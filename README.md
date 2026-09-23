# RTX 5070 fused INT4 weight-only GEMM

本工程针对 NVIDIA GeForce RTX 5070（`sm_120`）和固定
`M=N=K=4096`，实现并实测：

```text
C[M,N] = A_fp16[M,K] x dequant(B_int4[K,N], scale)
accumulator = FP32, default output = FP16
```

最终实现不 materialize 完整 FP16 B。B 在离线阶段量化并重排为 lane-native
MMA fragment；在线 mainloop 中才做 INT4 unpack、乘 group scale，并立即送入
`mma.sync.m16n8k16`。最终入口按 group 选择：

- group 32：V18a，96-register compact fragment-pair TMA；
- group 64：V17b，两 stage TMA + `ldmatrix.x4` A load；
- group 128：V18b，96-register compact pair + full/empty TMA；
- CTA `64x128x32`，warp `64x32`，4 warps / 128 threads；
- 全部 96 registers/thread，12,816/12,832 B dynamic SMEM，41.67% theoretical occupancy；
- 所有最终正式行 `max_abs_error=0`，NCU local spilling requests 为 0。

正式 30 warmup + 500 sample 结果位于
`results/final_v18_all96_e2e_formal_all_groups.csv`。V18 将 compact pair/full-empty/decode
的 32/64/128 九个实例全部压到 96 registers、零 spill，恢复 5 blocks/SM；最终 V18a
进入 group32，V18b 进入 group128，group64 保留 V17b。

## 环境

- RTX 5070，CC 12.0，48 SM，48 MiB L2，192-bit bus；
- CUDA Toolkit 13.0，`nvcc 13.0.88`，目标 `-arch=sm_120`；
- Driver 595.84，driver-supported CUDA 13.2；
- Nsight Compute 2025.3.1；
- Python 环境：Python 3.12.3 is used for benchmark and profiling helper scripts.；
- 完整设备属性：`results/hardware.txt`。
- Tested with CUDA Toolkit 13.0 / nvcc 13.0.88. The compiler must support sm_120.

本项目的 workload 是普通 signed INT4 + FP16 group scale，不以 NVFP4 结果替代。

## 构建与运行

```bash
make -j CUDA_HOME=/usr/local/cuda

./build/int4_gemm --correctness --size 256 --group 32
./build/int4_gemm --correctness --size 256 --group 64
./build/int4_gemm --correctness --size 256 --group 128
./build/int4_gemm --fp32-output-check --size 256 --group 32

./build/int4_gemm --benchmark --size 4096 --group all \
  --warmup 30 --iters 500 --only final \
  --csv results/final_v18_all96_e2e_formal_all_groups.csv

make ptx sass CUDA_HOME=/usr/local/cuda
python3 tools/parse_ncu.py results/profile/*.ncu-rep \
  --output results/profile/summary.csv
```

FP32 输出仅用于 correctness，可用 `make FP32_OUTPUT=0` 关闭。CMake 也支持
`INT4_GEMM_ENABLE_FP32_OUTPUT=OFF`。

Makefile 默认固定 `/usr/local/cuda-13.0`，避免父 shell 的 CUDA 12.8 `CUDA_HOME`
污染构建；命令行仍可显式覆盖。

## 编码与最终物理布局

原始 B 为 row-major signed INT4：

```text
logical = k*N + n
even logical -> low nibble; odd logical -> high nibble
q = sign_extend_4bit(nibble), q in [-8, 7]
scale = scales[(k/GROUP_SIZE)*N+n]
B_dequant[k,n] = FP16(q * scale)
```

早期版本先把 B 重排为 `[N/128][K/32][32][128]`，让每个 CTA 的 2 KiB
INT4 tile 物理连续。V16 再进一步把这 2 KiB 排成 MMA lane 的直接消费顺序：

```text
tile = nt*(K/32) + kt
word = [k16=2][warp_n=2][n8=8][lane=32]

n  = nt*128 + warp_n*64 + n8*8 + lane/4
k0 = kt*32 + k16*16 + (lane%4)*2

uint16 word = {
  q[k0,   n], q[k0+1, n], q[k0+8, n], q[k0+9, n]
}
```

一个 `uint16` 的四个 nibble 正好是一个 lane 向一次
`mma.sync.m16n8k16` 提供的四个 B 操作数；一个 32x128 tile 是 1024 个
word，仍为 2048 bytes。GPU repack 有独立 CPU mapping 校验；旧 tile layout
还有 GPU inverse-repack 逐字节校验。

group32/128 final 使用 fragment-pair layout：

```text
word32 = [warp_n=2][n8=8][lane=32]
low16  = k16 0 fragment
high16 = k16 1 fragment
```

因此一个 `uint32` shared load 同时取得同一 lane 的两个 k16 fragment；总 payload
仍是 2048 B/tile。group32 V18a 只跨 k16 缓存其中两个 pair，另外两个在使用点按
`uint16` 读取；group64 final 继续消费上面的普通 `uint16` layout。

scale 排成：

```text
scale_repacked[((n/128)*(K/group) + k/group)*128 + n%128]
```

所以一个 CTA 的 128 个 scale 也是连续的。4096² repacked weight 固定为
8,388,608 bytes；scale 大小为：

| Group | Scale bytes |
| ----: | ----------: |
|    32 |   1,048,576 |
|    64 |     524,288 |
|   128 |     262,144 |

相比 row-packed B，最终热循环不再为每个 nibble 计算 `k*N+n`，也不再把
完整 32x128 B tile 解码成 FP16 shared tile。地址在 preprocessing 阶段一次性
变换，在线阶段以 `tile_base + mma_fragment_word(...)` 直接取 16-bit word。

每个对齐 16-byte（128-bit）chunk 含 8 个连续 lane-word，也就是同一
`k16/warp_n/n8` 下连续 8 lanes 的 32 个 INT4 operand。B payload 本身仍是
2048 B/tile，即理论下限 64 个 32-byte sector；repack 不声称减少必需字节数。
row-packed tile 是 32 个彼此相距 `N/2=2048 B` 的 64-byte span，最多覆盖 32 个
分散的 128-byte line；最终是 16 个连续 128-byte line。cp.async 路径由 CTA 发出
128 次 16-byte B copy，TMA 路径则由一个 `UTMALDG.2D` 逻辑 tile load 提交 B。
本次 NCU section 没有保存精确 sector-request counter，因此只报告上述静态范围和实测
L2/DRAM 比率，不虚构“实际 transaction 减少量”。

## 最终 kernel 数据流

```text
GDDR/L2 A FP16 ----------- TMA ----------> 2-stage A SMEM -- ldmatrix.x4 A fragment --+
                                                                                      |
GDDR/L2 B INT4 --- offline lane repack ---> contiguous 2 KiB CTA tile                |
                  -------- TMA -----------> packed B SMEM -> uint16/uint32 per lane   |
                                                        -> f16x2 bit-magic decode     |
GDDR/L2 scale ----------------------------> scale SMEM ----> f16x2 multiply ----------+
                                                                                      |
                              mma.sync.m16n8k16 -> FP32 register accumulators --------+
                                                        -> direct half2 global store
```

INT4 转 FP16 使用 packed `f16x2` bit trick：先 XOR sign bit，将 nibble 嵌入
1024h，减 1032h，再乘广播的 half2 scale；避免标量 I2F/F2FP。累加结果按 PTX
lane mapping 直接 `half2` 写出，不再经过 32 KiB CTA-wide FP32 epilogue scratch。

### group 32：TMA V18a compact pair

Host 用 `cuTensorMapEncodeTiled` 建立三张 tensor map：

- A：64x32 FP16，4096 B；
- B：把 fragment-packed 2 KiB tile 表示为 16 个 128 B row；
- scale：128 FP16，表示为 2 个 128 B row。

一个线程发出三次 `cp.async.bulk.tensor.2d`，mbarrier 跟踪每 stage 的 6400 B
transaction；consumer CTA 等待 parity 后执行 MMA。两 stage 共 12,816 B。
SASS 中对应真实 `UTMALDG.2D`，不是普通 LDG 的包装。warp tile 改为 `64x32` 后，
A fragment 用 `ldmatrix.sync.aligned.m8n8.x4`，A-side 静态 LDS 从 v17a 的 48 降到 24。

V18a 在 pair layout 上只跨 k16 保留两个 `uint32` B pair，另外两个 fragment 在使用点
以 `uint16` 读取；四个广播 scale 收敛为两个 packed scale register。块坐标只在 TMA
发射和 epilogue 现场重算，不再跨 mainloop 保活。最终 group32 为 96 registers、零 spill，
恢复 5 blocks/SM。

### group 64：TMA V17b

TMA 形状、`64x32` warp tile 和 `ldmatrix.x4` 与 group32 相同，但交错正反顺序 A/B
显示 V18a 慢 1.05%，所以保留普通 MMA-fragment layout 的 V17b。

### group 128：TMA V18b compact full/empty

数据搬运和 A load 与 v17b 相同，B 使用 fragment-pair layout；V18 将 scale-row 地址按
group 在编译期化简，把 loader 整数状态移出长活跃区间，并使用 compact pair/scale 与
full/empty mbarrier。它从 V17c 的 100 registers、4 blocks/SM 降到 96 registers、
5 blocks/SM，动态 SMEM 为 12,832 B。交错正反顺序 A/B 比 V17c final 快 4.77%。

### TMA 与 TMEM

TMA 和 TMEM 是两种独立能力。RTX 5070 的 SM120 可执行 TMA；本工程 PTX/SASS
已实证。SM100 教程中的 `tcgen05.mma` 将 accumulator 写入 TMEM，而 NVIDIA 的
SM120 GeForce GEMM 路径使用扩展 MMA，不能把 SM100 的可编程 TMEM 路径直接套到
SM120。这个区别也与 CUTLASS 的
[SM120 GeForce examples](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm)、
[SM120 TMA collective](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm120_mma_tma.hpp)
及 [SM100 TMEM tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu)
的架构分支一致。

## GitHub 实现对照后采用的优化

对照 CUTLASS、DeepGEMM 和 FlashAttention 的开源实现后，优先验证了共同的高性能
模式：consumer-native layout、异步多 stage 搬运、mbarrier、swizzle、专用 DMA warp、
寄存器 epilogue 和 shape autotuning。这里没有直接照搬面向 SM90/SM100 或 FP8/NVFP4
的 kernel；每个想法都重新在 SM120 + 本项目 INT4-dequant workload 上实测。

- 采用：MMA-lane-native repack、register decode/MMA/direct store、两 stage TMA、
  `64x32` warp tile、`ldmatrix.x4`、group32 compact pair 与 group128 compact full/empty；
- 否决：TMA swizzle、3/4 stage、额外 DMA warp、16x64 warp tile；
- V18a compact pair 通过三组 correctness、正式 sweep 与交错 A/B，进入 group32 final；
- 32x32 warp tile 在 4096³ 出现稀疏数值错误，移出 registry。

参考仓库：[NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)、
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、
[FlashAttention](https://github.com/Dao-AILab/flash-attention)。

## Benchmark 规则

- CUDA Event；最终验证 30 warmup + 500 samples；排名看 median；
- 报告 median / mean / min / P95；
- 每个 timed kernel 先做完整输出 correctness gate；
- steady-state 区间不包含 allocation、quantization 或 repack；
- cold-ish 在每个 GEMM Event 前运行 128 MiB cache-disturb，disturb 本身不计时；
- one-shot end-to-end 在同一 Event 内执行 quantize + 最终 weight/scale repack + final GEMM；
- effective TFLOP/s = `2*M*N*K/latency`，不把 dequant 指令算作额外 FLOP。

cuBLAS 行的 B 已完整 materialize 为 32 MiB FP16，和 fused INT4 workload 不等价，
只作为硬件参考。正式运行存在 DVFS 波动，因此只对同一次进程中的行做严格百分比比较。

## 最终 4096³ 结果

来源：`results/final_v18_all96_e2e_formal_all_groups.csv`，30/500，无并发 GPU 进程。

| Group | Quantize | Final repack (B+scale) | One-shot end-to-end | Separate dequant+cuBLAS |     Final steady |         Mean / Min / P95 |          TFLOP/s | Final cold-ish |
| ----: | -------: | ---------------------: | ------------------: | ----------------------: | ---------------: | -----------------------: | ---------------: | -------------: |
|    32 |   1.0723 |            0.0334 pair |              3.7298 |                  2.4951 | **2.4035** | 2.3878 / 2.1454 / 2.6102 | **57.183** |         2.3561 |
|    64 |   0.5741 |                 0.0396 |              3.1444 |                  2.4767 | **2.4128** | 2.4055 / 2.1555 / 2.6257 | **56.962** |         2.3692 |
|   128 |   0.4135 |            0.0314 pair |              2.8462 |                  2.5019 | **2.3770** | 2.3785 / 2.1425 / 2.6051 | **57.820** |         2.3702 |

同次运行中，final 相对 separate dequant+cuBLAS 在 group 32/64/128 分别快
3.81%/2.65%/5.25%。cold-ish 略快于 steady 属于噪声，不能解释为冷缓存加速。

V17 与旧 final 的交错正反顺序 A/B 更适合衡量代码收益：group32/64 的 v17b 平均快
3.45%/2.71%，group128 的 v17c 快 2.07%。对应原始文件为 `results/ab_v17_*.csv`；
`results/v17_formal_all_groups.csv` 保存 v17a--e 与旧 final 的同场 20/200 sweep。

第一轮 V18a 对 V17 final 的顺序控制为 group32 **+0.86%**，因此更新 group32。scale-row
编译期化简把所有 group 的 V18 实例进一步压到 96 regs 后，group64 V18a 与 V17b 持平
（+0.001%），group128 V18b 对 V17c 为 **+4.77%**；因此 group64 不换，group128 更新为
V18b。原始文件为 `results/ab_v18_g*_*.csv` 与 `results/ab_v18_all96_g*_*.csv`。

V16 前的正式 `results/benchmark.csv` 中旧 final 为 2.6825 / 2.8729 / 2.8741 ms。
新正式 run 的绝对数字明显更快，但跨 run 会受频率影响；版本收益判断使用下列同场
候选文件，而不是把两个 run 的最好值相除。

### V16 同场增益

| 实验              | Group 32 | Group 64 | Group 128 | 结论                            |
| ----------------- | -------: | -------: | --------: | ------------------------------- |
| V16a vs old final |   +9.89% |  +13.04% |   +14.17% | 低 SMEM warp epilogue 有效      |
| V16b vs V16a      |   +5.40% |  +13.81% |   +16.01% | lane repack + register MMA 有效 |
| V16c0 TMA vs V16b |   +2.94% |   +1.49% |    -0.13% | TMA 只选 g32/g64                |

对应文件：`v16a_formal_all_groups.csv`、`v16b_formal_all_groups.csv`、
`v16c0_formal_all_groups.csv`。V16 finalist 同场 sweep 位于
`v16_finalists_formal_all_groups.csv`。

### group 32 优化时间线

| Version   | Optimization                          |        Median ms | Effective TFLOP/s |                                 相比上一有效阶段 |
| --------- | ------------------------------------- | ---------------: | ----------------: | -----------------------------------------------: |
| v00       | scalar fused baseline                 |         234.3499 |             0.586 |                                         baseline |
| v01       | pairwise INT4 + half2                 |         151.8928 |             0.905 |                                            1.54x |
| v02       | tile-contiguous repack                |         109.8615 |             1.251 |                                            1.38x |
| v03       | WMMA Tensor Core                      |           5.3245 |            25.812 |                                           20.63x |
| v04b      | repacked CTA search                   |           5.0402 |            27.269 |                                            1.06x |
| v06       | 32x64 warp tile                       |           4.7065 |            29.202 |                                            1.07x |
| v09s2     | 2-stage cp.async                      |           4.0547 |            33.896 |                                            1.16x |
| v11d      | shared padding winner                 |           2.6719 |            51.439 |                                            1.52x |
| v16a      | warp epilogue                         |           2.6554 |            51.758 |                     +9.89% vs same-run old final |
| v16b      | lane repack/register MMA              |           2.5650 |            53.583 |                          +5.40% vs same-run v16a |
| v16c0     | full TMA register MMA                 |           2.4460 |            56.190 |                          +2.94% vs same-run v16b |
| V16 final | g32 V16c0, independent formal run     | **2.2228** |  **61.832** | architecture same as v16c0; DVFS-sensitive rerun |
| v17b      | 64x32 + ldmatrix TMA, interleaved A/B |       2.3990 avg |             57.29 |                     +3.45% vs same-run old final |
| v18a      | compact fragment-pair, 96 regs        |   2.5255 A/B avg |             54.42 |                     +0.86% vs same-run V17 final |

早期 v00--v11d 来自原始同一 formal run；V16 每行的改进百分比来自各自同场文件。
不能把跨 run 的 final 2.2228 与 v16c0 2.4460 当作代码带来的 10% 提升。

## Nsight Compute

机器可读汇总为 `results/profile/summary.csv`。NCU replay duration 只用于 profiler
内比较，不与 CUDA Event latency 混用。

| Metric                   |    old v11d g32 | V16 final TMA g32 | V16 final cp.async g128 |
| ------------------------ | --------------: | ----------------: | ----------------------: |
| NCU duration ms          |          3.1878 |  **2.6874** |                  2.7396 |
| Tensor/SM active         |          37.87% |  **44.93%** |                  44.07% |
| DRAM throughput          |           3.46% |             4.74% |                   4.87% |
| L2 hit                   |          95.63% |            96.51% |                  96.83% |
| Active warps/SM          |           11.73 |   **19.23** |                   19.21 |
| Eligible warps/scheduler |           0.413 |             0.492 |         **0.574** |
| Long scoreboard          |           0.844 |   **0.053** |                   0.156 |
| Short scoreboard         |           1.870 |   **0.234** |                   0.260 |
| Barrier                  | **0.191** |             2.611 |                   1.186 |
| Math pipe throttle       |           1.710 |             7.492 |                   7.523 |
| Registers/thread         |              94 |                96 |                      94 |
| Dynamic SMEM KiB         |          32.768 |  **12.816** |                  14.848 |
| Achieved occupancy       |          24.44% |  **40.07%** |                  40.03% |
| Local spill requests     |               0 |                 0 |                       0 |

V16 的核心收益是去掉 decoded-B 和 CTA-wide epilogue shared tile，将 achieved
occupancy 从 24.44% 提升到约 40%，并显著压低 long/short scoreboard。TMA g32 的
barrier stall 反而升到 2.61，说明 mbarrier/CTA sync 仍有成本。DRAM throughput 只有
约 5%、L2 hit 约 97%，最终不是 DRAM-bound；新的主要瓶颈已经转为 packed `f16x2`
decode/scale 和相关地址/发射带来的 Math Pipe Throttle（约 7.5），其次是 TMA barrier。

### V17 NCU 裁决

| Metric             |   v17a |             v17b |   v17c |            v17d |            v17e |
| ------------------ | -----: | ---------------: | -----: | --------------: | --------------: |
| Duration ms        | 2.6317 | **2.6114** | 2.6753 |          2.6790 |          2.6838 |
| Tensor active      | 45.89% | **46.25%** | 45.12% |          45.05% |          44.98% |
| Math Pipe          |  9.921 |           11.190 |  7.855 |           7.278 | **6.410** |
| Barrier            |  3.466 |            3.430 |  3.165 | **0.003** | **0.003** |
| Achieved occupancy | 39.90% |           39.87% | 32.29% |          32.24% |          32.24% |

`ldmatrix.x4` 让 v17b 的静态 A-side LDS 48→24，并得到最快 group32 NCU duration。
pair/scale 跨 k16 保活使 v17c--e 从 96 增至 98 registers，跨过 5→4 blocks/SM 台阶；
因此 v17d 的 barrier 清零、v17e 的 Math Pipe 降低都没有形成净吞吐收益。group128 则由
pair reuse 获益：v17c NCU 2.6807 ms，对旧 final 的 2.7445 ms 快 2.38%。

### V18 all-group 96-register NCU 裁决

| Metric                     | v18a compact pair | v18b full/empty |   v18c BFE/PRMT |
| -------------------------- | ----------------: | --------------: | --------------: |
| Registers / spill requests |            96 / 0 |          96 / 0 |          96 / 0 |
| Duration ms                |  **2.5996** |          2.6036 |          2.6179 |
| Tensor active              |  **46.43%** |          46.39% |          46.13% |
| Active warps/SM            |             19.14 |           19.11 |           19.12 |
| Achieved occupancy         |            39.88% |          39.82% |          39.84% |
| Barrier                    |             3.398 | **0.003** | **0.003** |
| Math Pipe                  |            11.452 |          11.353 | **9.835** |

两枚 packed scale register、按使用点加载半数 B fragment、现场重算 block 坐标、32-bit
shared mbarrier wait，以及按 group 编译期化简的 scale-row 共同把 32/64/128 九个 V18
实例全部压到 96 registers，且没有依赖会产生 spill 的 `launch_bounds`。group32 的 v18b/c
虽改善目标指标，仍由 v18a 最快。

最终 selected profile 中，group32 v18a / group128 v18b 的 duration 为
2.6064/2.6040 ms，Tensor active 46.32%/46.37%，achieved occupancy 39.88%/39.82%，
local spill requests 均为 0；group128 v18b 的 Barrier 为 0.003。机器可读数据在
`results/profile/v18_all96_selected_summary.csv`。

## PTX / SASS 证据

- `results/v16_register.ptx`：`cp.async.ca.shared.global`、
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`；
- `results/v16_tma.ptx`：`cp.async.bulk.tensor.2d...mbarrier`、
  `mbarrier.arrive.expect_tx`、`mbarrier.try_wait.parity`；
- `results/final.sass`：final TMA instance 有 `UTMALDG.2D`，两条路径均有
  `HMMA.16816.F32`；cp.async 路径对应 `LDGSTS.E.128`；
- cuobjdump/ptxas：cp.async final `LOCAL:0`，TMA tensor-map 参数有 16 B stack frame，
  但 `LOCAL:0`、NCU local spilling requests=0，二者都无 register spill。
- PTX/SASS 为构建生成文件，因此不提交到 Git 仓库。可通过 `make ptx sass` 重新生成。实机检查确认 final TMA 路径包含 `UTMALDG.2D`、`HMMA.16816.F32`，cp.async 路径包含 `LDGSTS.E.128`；ptxas 与 NCU 均未检测到 register spill。
- 原始 `.ncu-rep` 文件体积较大且包含本机 profiler 元数据，因此未提交；仓库保留机器可读的 NCU summary CSV。

## 保留的失败实验

- 128x128 CTA、BK64/BK128：SMEM/register 增长超过复用收益；
- 3/4-stage cp.async：没有稳定超过两 stage；
- persistent 128-CTA：group32 比 normal 慢 21.18%；
- 64x64 warp tile：202 registers/thread、12.5% occupancy，慢约 50%；
- V15 TMA + WMMA：搬运减少但 decoded-B/epilogue SMEM 未消除，没有稳定收益；
- V16c swizzle：寄存器升到约 126--128，抵消 bank-layout 收益；
- V16c 3/4 stage：19,224/25,632 B SMEM 和额外同步使延迟回退；
- V16c DMA warp：160 threads 和控制开销回退；
- V16d1 16x64：重复 B decode，group32 仅约 49.5 TFLOP/s；
- V16d0 32x32：4096³ 稀疏数值错误，拒绝；
- V16d2 64x32：很快但一次正式 group64 correctness gate 失败，可靠性不足，拒绝。

完整 hypothesis、修改、数据与结论见 `OPTIMIZATION_LOG.md`。

## 结论与剩余空间

最终 kernel 快的原因不是更高 DRAM 带宽，而是把离线布局做到 MMA lane-native，在线
只搬 packed INT4，使用寄存器内 f16x2 decode/scale、显式 MMA 和直接 half2 epilogue。
V17 再用 `64x32` warp tile 提高 B decode reuse，以 `ldmatrix.x4` 降低 A-side LDS；V18
将三种 group 的 compact pair/full-empty/decode 全部压到 96 registers。三种 group 最终
都使用两 stage TMA，group32 使用 compact pair，group128 使用 compact full/empty，
group64 保留普通 fragment layout。

寄存器 occupancy cliff 已解决；下一步应减少 compact pair 的 decode/scale issue，或
寻找不增加寄存器和 warp 指令的 full/empty handoff。也可用 CUTLASS/CuTe 的 SM120
collective 做等 workload 对照。TMEM 不是 RTX 5070 可用的 SM100 `tcgen05` 路径。

## 目录

```text
include/      encoding, layout, benchmark and launch interfaces
src/          quantize, repack, CPU reference and benchmark harness
kernels/      independently runnable optimization versions
tools/        benchmark/autotune/NCU parsing
results/      formal CSV, hardware, PTX/SASS and NCU reports
```
