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

## 性能优化全过程与中间结果（Detailed Optimization Journey

---

### 1. 总体性能演进

| 版本     | 当时主要缺陷                                  | 解决手段                          |                       Median latency |                 Effective TFLOP/s | 结果                                  |
| -------- | --------------------------------------------- | --------------------------------- | -----------------------------------: | --------------------------------: | ------------------------------------- |
| V00      | 纯标量 SIMT，Tensor Core 完全未使用           | 建立 scalar fused baseline        |                          234.3499 ms |                             0.586 | 基线                                  |
| V01      | INT4 nibble / 地址 / store 指令重复           | pairwise INT4 +`half2`          |                          151.8928 ms |                             0.905 | 1.543x                                |
| V02      | B 在 K 方向跨行 gather，tile 不连续           | CTA tile-contiguous repack        |                          109.8615 ms |                             1.251 | 1.383x                                |
| V03      | SIMT FMA 吞吐成为绝对瓶颈                     | WMMA Tensor Core                  |                            5.3245 ms |                            25.812 | 20.633x                               |
| V04      | CTA shape 未调优，资源/复用不平衡             | CTA tile search                   |                            5.0402 ms |                            27.269 | +5.6%                                 |
| V05      | 每 BK32 都同步，尝试减少 barrier              | BK64 / BK128                      |                       5.6732 ms best |                            24.226 | 回退                                  |
| V06      | Warp 内 fragment reuse 不足                   | `32x64` warp tile               |                            4.7065 ms |                            29.202 | +20.5% vs V05b                        |
| V07      | 假设 B cache reuse 不够                       | B-reuse grid mapping              |                            4.7884 ms |                            28.702 | -1.74%                                |
| V08      | 验证更多 M 方向 reuse                         | 128x64 CTA                        |                       5.1610 ms best |                            26.631 | 回退                                  |
| V09      | Long Scoreboard / Barrier 高                  | 2-stage `cp.async`              |                            4.0547 ms |                            33.896 | +16.1% vs V06                         |
| V10      | Short Scoreboard 高，shared operand path 串行 | A/B shared padding                |                            2.6754 ms |                            51.372 | 约 1.52x                              |
| V11      | padding 仍未收敛                              | A+8 / B+24                        |                            2.6719 ms |                            51.439 | 第一阶段 winner                       |
| V12      | 判断 padding 是否还有空间                     | stopping sweep                    |                          ≥2.6812 ms |                          ≤51.261 | 停止                                  |
| V13      | 假设 CTA scheduling / launch overhead         | persistent CTA                    |                            3.2660 ms |                            42.082 | 慢 21.18%                             |
| V14      | 旧路径 warp shape 可能未搜索完整              | 64x32 / 64x64                     |                   2.7743 / 4.0485 ms |                   49.541 / 33.948 | 均回退                                |
| V15      | cp.async copy issue 仍占发射槽                | 第一轮 SM120 TMA                  |                       best 2.9995 ms |                            45.821 | 旧路径无稳定收益                      |
| V16a     | 32 KiB SMEM 压低 occupancy                    | low-SMEM warp epilogue            |                           2.6554 ms* |                            51.758 | g32 +9.89%                            |
| V16b     | decoded FP16 B shared tile 成为主要负担       | lane-native repack + register MMA |                           2.5650 ms* |                            53.583 | g32 +5.40%                            |
| V16c0    | register path 中 copy issue 重新成为问题      | 2-stage TMA register MMA          |                           2.4460 ms* |                            56.190 | g32 +2.94%                            |
| V16d2    | 数据流改变后旧 warp-shape 结论失效            | `64x32` retune                  |                           2.4906 ms* |                               ~55 | 有竞争力                              |
| V17b     | 64x32 后 A-side scalar LDS 增多               | `ldmatrix.x4`                   |                           2.3675 ms* |                            58.053 | g32 winner                            |
| V17c     | B decode / scale reuse 不足                   | fragment-pair repack              |                      2.4444 ms g128* |                            56.226 | g128 winner，但 g32 触发 98-reg cliff |
| V17d     | TMA CTA-wide barrier 仍明显                   | full/empty mbarrier               |                        NCU 2.6790 ms |                                — | Barrier 几乎清零，但总延迟没赢        |
| V17e     | Math Pipe 仍高                                | BFE/PRMT decode                   |                        NCU 2.6838 ms |                                — | Math Pipe 降低，但总延迟没赢          |
| V18a/b/c | 98/100 regs 导致 5→4 CTA/SM                  | compact live-range，压回 96 regs  |              ~2.36–2.42 ms same-run |                           ~57–58 | 恢复 5 CTA/SM                         |
| Final    | decode / scale / issue 成为主要剩余瓶颈       | group-aware dispatch              | 2.4035 / 2.4128 /**2.3770** ms | 57.183 / 56.962 /**57.820** | 当前最终                              |

\* 对应阶段的 same-run / formal candidate 数据。
最终代码收益优先以同场 A/B 为准，不把跨 run 的绝对最低值直接相除。

整体瓶颈迁移可以概括为：

```text
SIMT compute
    ↓ Tensor Core
Global / barrier latency
    ↓ cp.async
Shared-memory operand dependency
    ↓ padding / layout
SMEM footprint / occupancy
    ↓ low-SMEM epilogue
Decoded-B shared path
    ↓ lane-native register MMA
Copy issue / A-side LDS
    ↓ TMA + ldmatrix.x4
Register occupancy cliff
    ↓ compact live-range
INT4 decode / scale / instruction issue
    ↓ 当前主要剩余瓶颈
```

---

## 2. V00 — Scalar Fused Baseline

### 当前缺陷 / 瓶颈

项目最开始没有自定义 fused kernel，需要先确认：

```text
INT4 packing 是否正确
sign-extension 是否正确
group scale 索引是否正确
fused dequant + GEMM 的数值语义是否正确
```

因此 V00 有意采用最直接的实现：每个 thread 计算一个 C 元素。

### 怎么发现性能问题

V00 的资源并不紧张：

```text
38 registers/thread
0 dynamic SMEM
Achieved occupancy ≈ 99.82%
```

但性能只有：

```text
234.3499 ms
0.586 TFLOP/s
Tensor active = 0
```

这说明问题不是 occupancy，而是：

> **纯 CUDA Core 标量 dot-product 的算术吞吐远远不够。**

### 为什么会慢

每个 output 元素都需要：

```text
逐个读取 packed B
逐个拆 nibble
逐个 sign extend
逐个 scale
逐个 scalar FMA
```

同时完全没有使用 Tensor Core。

### 解决方式

V00 本身不追求性能，只负责建立：

```text
正确性基线
性能基线
后续 fused kernel 的对照
```

### 性能

```text
234.3499 ms
0.586 TFLOP/s
```

### 解决后还剩什么问题

核心问题完全没有解决：

```text
SIMT FMA throughput 太低
INT4 pair 没被高效消费
地址与 nibble 操作冗余
```

因此下一步先做最便宜的指令级优化 V01。

### 源码 / 数据

```text
kernels/gemm_v00_reference.cu
results/benchmark.csv
results/profile/summary.csv        # v00_g32
```

---

## 3. V01 — Pairwise INT4 Load + `half2`

### 当前缺陷 / 瓶颈

V00 把 packed INT4 当成“单个权重”处理，但实际上：

```text
1 byte = 2 × INT4
```

因此出现：

```text
同一个 byte 被重复解析
nibble unpack 重复
地址计算重复
输出 store 粒度太细
```

### 怎么判断

虽然 V00 occupancy 很高，但吞吐仍只有 0.586 TFLOP/s，说明应该先降低每个逻辑 MAC 周围的指令开销。

### 解决方式

将处理粒度改成 INT4 pair：

```text
一次 load 一个 packed byte
一次拆出两个 INT4
一次性生成两个半精度结果
half2 store
```

### 性能

```text
234.3499 → 151.8928 ms
0.586 → 0.905 TFLOP/s

Speedup = 1.543x
Latency -35.19%
```

### 为什么有效

减少了：

```text
global load 指令
nibble unpack 次数
地址计算
store 指令数
```

### 解决后还剩什么问题

即便降低了指令冗余，CTA 读取 B 时仍然沿 row-major 跨 K 行 gather：

```text
相邻 K 行物理间隔 = N/2 = 2048 B
```

因此下一步转向内存布局。

### 源码 / 数据

```text
kernels/gemm_v01_pairwise.cu
results/benchmark.csv
```

---

## 4. V02 — CTA Tile-Contiguous Memory Repack

### 当前缺陷 / 瓶颈

V01 仍直接消费 row-packed B。

对一个 CTA 需要的 `32×128` B tile 来说，不同 K 行位于分散的 global memory 区域。

### 为什么会慢

Hot loop 中需要频繁：

```text
k*N+n
跨 K 行跳转
非连续 tile gather
复杂地址计算
```

即使 packed INT4 总数据量很小，**访问模式仍不适合后续向量搬运和 Tensor Core staging**。

### 解决方式

在 GEMM 之前增加 offline repack：

```text
row-packed B
→
[N/128][K/32][32][128]
```

让一个 CTA 的：

```text
BK×BN = 32×128
```

成为连续的：

```text
2048 B
```

Scale 同时按 CTA tile 重排。

并增加：

```text
GPU repack
GPU inverse-repack
CPU mapping reference
```

保证 layout transformation 正确。

### 性能

```text
151.8928 → 109.8615 ms
0.905 → 1.251 TFLOP/s

Speedup = 1.383x
Latency -27.67%
```

### 为什么有效

减少了 runtime gather 与地址计算，并为后续：

```text
vector load
cp.async
TMA
warp/lane-native repack
```

奠定物理布局基础。

### 解决后还剩什么问题

SIMT compute 仍然是绝对瓶颈。

即便数据更连续，1.251 TFLOP/s 仍然远低于 Tensor Core 能力。

所以 V03 必须改变计算单元。

### 源码 / 数据

```text
kernels/gemm_v02_repack_simt.cu
src/repack.cu
results/benchmark.csv
```

---

## 5. V03 — 首次 Tensor Core 路径

### 当前缺陷 / 瓶颈

V02 已经证明 layout 有收益，但：

```text
109.8615 ms
1.251 TFLOP/s
```

说明继续优化 SIMT load 不能解决主问题。

### 怎么判断

此阶段主要瓶颈是：

> **算术吞吐，而不是内存字节数。**

### 解决方式

引入第一版 Tensor Core：

```text
CTA tile   = 64×128×32
Warp tile  = 32×32
Accumulator = FP32
```

在线数据流：

```text
INT4 B
→ unpack
→ × scale
→ FP16 shared tile
→ WMMA m16n16k16
```

### 性能

```text
109.8615 → 5.3245 ms
1.251 → 25.812 TFLOP/s

Speedup = 20.633x
Latency -95.15%
```

这是全项目最大的一次单步性能跃迁。

### 新暴露的问题

NCU：

```text
Tensor active       19.14%
Long Scoreboard      9.14
Short Scoreboard     5.14
Barrier              1.77
MIO                  2.67
```

虽然 Tensor Core 已经启用，但利用率仍然不高。

新的问题变成：

```text
global → shared latency 没隐藏
shared operand load 存在依赖
barrier 频繁
```

### 下一步为什么做 V04/V05

先搜索 CTA/BK，确定是 tile 配置问题还是 pipeline 问题。

### 源码 / 数据

```text
kernels/gemm_v03_tensorcore.cu
results/benchmark.csv
results/profile/summary.csv        # v03_g32
```

---

## 6. V04 — CTA Tile Search

### 当前缺陷 / 瓶颈

V03 的 `64×128` CTA 是初始选择，并没有通过数据验证。

需要判断：

```text
M reuse
N continuity
SMEM
register
occupancy
```

之间的平衡。

### 解决方式

测试：

```text
64×128
128×64
128×128
```

### 性能

| CTA      |              Median |          TFLOP/s | Registers |   SMEM | Occupancy |
| -------- | ------------------: | ---------------: | --------: | -----: | --------: |
| 64×128  |           5.2368 ms |           26.245 |        62 | 32 KiB |       50% |
| 128×64  | **5.0402 ms** | **27.269** |        72 | 32 KiB |       50% |
| 128×128 |           8.1366 ms |           16.891 |        64 | 64 KiB |    33.33% |

### 为什么 128×128 失败

更大的 tile 虽然理论 reuse 更高，但：

```text
SMEM 32 KiB → 64 KiB
Occupancy 50% → 33.33%
```

residency 损失远大于 reuse 收益。

### 解决后还剩什么问题

即使选出更合适 CTA，V03 NCU 中的 barrier / scoreboard 仍然存在。

因此下一步测试增大 BK 是否能减少同步。

### 源码 / 数据

```text
kernels/gemm_v04_repack_tensorcore.cu
results/benchmark.csv
results/profile/summary.csv        # v04a_g32
```

---

## 7. V05 — BK64 / BK128：减少 Barrier 的失败尝试

### 当前缺陷 / 瓶颈

每个：

```text
BK = 32
```

都需要一次 staging 与同步。

直觉上 BK 增大可以减少 K-loop barrier 数量。

### 解决方式

测试：

```text
BK = 64
BK = 128
```

### 性能

|  BK |    Median | TFLOP/s | Registers |   SMEM | Occupancy |
| --: | --------: | ------: | --------: | -----: | --------: |
|  64 | 5.7374 ms |  23.955 |        70 | 32 KiB |       50% |
| 128 | 5.6732 ms |  24.226 |        92 | 48 KiB |    33.33% |

### 为什么失败

虽然 barrier 次数减少，但在还没有 load/compute overlap 时，BK 增大会使：

```text
A/B tile live state ↑
register ↑
SMEM ↑
occupancy ↓
```

BK128 直接把：

```text
register = 92
SMEM = 48 KiB
occupancy = 33.33%
```

最终比 V04 更慢。

### 结论

> **不能只减少同步次数，而忽略资源 footprint。**

### 下一步

回到更小 BK，把优化重点转向 warp 内 fragment reuse，而不是继续扩大 K tile。

### 数据

```text
results/benchmark.csv
OPTIMIZATION_LOG.md / Version V05
```

---

## 8. V06 — `32×64` Warp Tile

### 当前缺陷 / 瓶颈

V05 说明扩大 BK 不合适，但当前 warp tile 仍存在：

```text
A fragment 复用不足
warp/CTA 数量较多
```

### 解决方式

将：

```text
Warp Tile: 32×32 → 32×64
```

CTA 使用：

```text
4 warps / 128 threads
```

同一个 A fragment 可以服务更多 B fragments / MMA。

### 性能

```text
5.6732 → 4.7065 ms
24.226 → 29.202 TFLOP/s

+20.5% vs V05b
```

### 一个重要现象

资源：

```text
100 registers/thread
32 KiB SMEM
25% theoretical occupancy
```

occupancy 反而下降，但性能提高。

这说明：

> 此阶段更高的数据复用比单纯追求高 occupancy 更重要。

### 解决后还剩什么问题

global→shared 仍然是同步路径，V03/V04 的 Long Scoreboard 问题并没有从根本解决。

在进入 async pipeline 前，先补做 grid locality 和 CTA 方向搜索。

### 源码 / 数据

```text
kernels/gemm_final.cu              # registry 中 v06
results/benchmark.csv
```

---

## 9. V07 — B-Reuse Grid Mapping：失败

### 当前假设

如果 B bandwidth / cache locality 是瓶颈，那么让相邻 CTA 使用相同 N tile，应该提高 B reuse。

### 解决方式

改变 block 到 M/N tile 的映射，让调度更偏向：

```text
same N tile
different M tile
```

### 性能

```text
4.7065 → 4.7884 ms
-1.74%
```

### 为什么失败

packed B 只有：

```text
8 MiB
```

而 RTX 5070 L2：

```text
48 MiB
```

steady-state 下 B 本来就很容易驻留 L2。

因此：

```text
额外 grid swizzle
```

并没有解决真实瓶颈，反而增加调度/索引成本。

### 结论

恢复标准 N-major grid。

### 下一步

验证另一个 CTA 轴向：128×64。

---

## 10. V08 — 128×64 CTA：失败

### 当前假设

增加 M 方向 tile 可以进一步提高 B reuse。

### 解决方式

测试：

```text
128×64×32
128×64×64
```

### 性能

```text
BK32: 7.5832 ms / 18.124 TFLOP/s
BK64: 5.1610 ms / 26.631 TFLOP/s
```

### 为什么失败

最高寄存器达到：

```text
122 registers/thread
```

更多 M reuse 没有抵消：

```text
register pressure
更差的 load mapping
更差的调度效率
```

另外一个 BN64 half-physical vector mapping 出现：

```text
relative L2 ≈ 3.08%
```

被 correctness gate 直接拒绝。

### 结论

继续保留 64×128 CTA。

### 下一步

此时 tile 方向基本确定，真正需要解决的是 global→shared latency，因此进入 `cp.async`。

### 数据

```text
results/benchmark.csv
results/probe_v08_g32.csv
```

---

## 11. V09 — 2-Stage `cp.async`

### 当前缺陷 / 瓶颈

NCU 已经明确暴露：

```text
Long Scoreboard 高
Barrier 高
MIO 高
```

说明当前执行顺序是：

```text
load
→ wait
→ compute
→ load
→ wait
→ compute
```

Global memory latency 没有被计算覆盖。

### 解决方式

建立 double-buffer pipeline：

```text
compute tile i
同时
prefetch tile i+1
```

测试：

```text
2 stage
3 stage
4 stage
```

### 性能

| Stage |              Median | TFLOP/s |
| ----: | ------------------: | ------: |
|     2 | **4.0547 ms** |  33.896 |
|     3 |           4.0832 ms |  33.659 |
|     4 |           4.0532 ms |  33.909 |

4-stage 的单次结果只快约 0.04%，没有稳定优势，因此保留 2-stage。

### 相比 V06

```text
4.7065 → 4.0547 ms
Latency -13.85%
Speedup ≈ 1.161x
```

### NCU 验证

```text
Long Scoreboard 9.14 → 1.52
Barrier         1.77 → 0.54
MIO             2.67 → 0.41
```

### 新暴露的问题

```text
Short Scoreboard → 6.61
```

也就是说 global latency 被隐藏后，主要问题迁移到：

> **Shared Memory → Tensor Core operand register 路径。**

### 下一步

针对 shared bank / stride 做 layout tuning。

### 源码 / 数据

```text
kernels/gemm_v09_async.cu
results/benchmark.csv
results/probe_async_g32.csv
results/profile/summary.csv        # v09s2_g32
```

---

## 12. V10 — Shared-Memory Padding

### 当前缺陷 / 瓶颈

V09 后 Long Scoreboard 已明显降低，但：

```text
Short Scoreboard = 6.61
```

成为最明显的 stall。

### 为什么会慢

这意味着 Tensor Core operand 在：

```text
SMEM → register
```

阶段存在较重依赖 / bank 访问串行。

继续优化 global pipeline 已经不是最高优先级。

### 解决方式

改变 shared stride：

```text
A +8
B +8
A +8 / B +8
```

### 性能

| Layout  |              Median |          TFLOP/s |
| ------- | ------------------: | ---------------: |
| A+8     |           3.5548 ms |           38.662 |
| B+8     |           2.7869 ms |           49.316 |
| A+8/B+8 | **2.6754 ms** | **51.372** |

### 相对 V09

```text
4.0547 → 2.6754 ms
约 1.516x
Latency -34.02%
```

### 为什么提升这么大

资源几乎没变，但 shared address mapping 改变后，operand path 的冲突/依赖显著下降。

这说明此前隐藏的真正问题不是 global bandwidth，而是 shared-memory layout。

### 新问题

padding 的最佳值未知，需要局部 sweep。

---

## 13. V11 — Padding Sweep

### 当前缺陷

V10 的 `A+8/B+8` 是人工选点，不能确认已经最优。

### 解决方式

测试：

```text
A+8/B+16
A+16/B+8
A+16/B+16
A+8/B+24
```

### 最佳

```text
A+8/B+24
2.6719 ms
51.439 TFLOP/s
```

### NCU

```text
Tensor active       37.87%
Long Scoreboard      0.844
Short Scoreboard     1.870
Barrier              0.191
MIO                  0.052
Math Pipe            1.710
```

### 说明

与 V09 相比：

```text
Short Scoreboard 6.61 → 1.87
```

验证了 shared-layout 假设。

### 新问题

V11 相比 V10 只再提高约 0.13%，说明 padding 搜索已经接近收敛。

因此 V12 不再追求“大改”，而是做停止条件验证。

---

## 14. V12 — Stopping Sweep

### 当前问题

需要确认：

> V11d 是真正局部最优，还是只是因为搜索范围太窄？

### 解决方式

继续测试：

```text
A+8/B+32
A+8/B+40
A+16/B+24
```

### 结果

没有任何候选稳定快于 V11d >= 1%。

### 结论

停止 padding 参数搜索。

这是项目中的一个重要原则：

> **优化停止由数据决定，而不是“感觉差不多了”。**

### 下一步

开始验证其他更高层次方向：persistent scheduling、warp shape、TMA。

### 数据

```text
results/benchmark.csv
results/probe_padding_g32.csv
results/probe_padding_sweep_g32.csv
results/padding_100_g32.csv
results/profile/summary.csv        # v11d_g32
```

---

## 15. V13 — Persistent CTA：失败

### 当前假设

如果当前性能受：

```text
kernel scheduling
CTA launch
跨 CTA cache locality
```

影响，那么 persistent CTA 可能更快。

### 解决方式

只发出：

```text
128 CTAs
```

每个 CTA grid-stride 处理多个 output tiles。

### 性能

group32：

```text
normal      2.6953 ms
persistent  3.2660 ms

Penalty +21.18%
```

group64 / 128 也分别慢：

```text
18.58%
18.36%
```

### 为什么失败

说明当前不是 launch/scheduling bound。

Persistent 反而增加：

```text
tile 间 barrier
长寿命 CTA
减少硬件调度自由度
```

### 结论

停止 persistent 搜索。

### 数据

```text
results/persistent_100.csv
```

---

## 16. V14 — 64×32 / 64×64 Warp Tile：旧路径下失败

### 当前问题

此前只重点使用 32×64，需要补全 warp-shape search。

### 解决方式

测试：

```text
64×32
64×64
```

### group32

```text
32×64 : 2.7002 ms
64×32 : 2.7743 ms   (-2.74%)
64×64 : 4.0485 ms   (-49.93%)
```

64×64：

```text
202 registers/thread
12.5% occupancy
```

### 为什么失败

64×64 accumulator 数量过多，引发极高 register pressure。

64×32 在**当时的 WMMA + decoded-B shared 数据路径**下也没有收益。

### 关键经验

这个结论只适用于当时的实现。

后续 V16 改成 register MMA 后，B reuse / register balance 都变了，因此 V16d 会重新测试 64×32。

### 数据

```text
results/warp_shapes_100.csv
```

---

## 17. V15 — 第一轮 SM120 TMA

### 当前缺陷 / 假设

V11 final 仍由 128 个线程发出大量 16-byte `cp.async`。

RTX 5070 / SM120 支持 TMA，因此假设：

> 一个线程提交整 tile transaction，可以减少 copy issue 指令。

### 解决方式

实现：

```text
cuTensorMapEncodeTiled
cp.async.bulk.tensor.2d
mbarrier.arrive.expect_tx
mbarrier.try_wait.parity
```

测试：

```text
v15a TMA B + cp.async A
v15b TMA A/B
v15c 3-stage
v15d 4-stage
v15e leader-wait
```

### 实现过程中遇到的真实问题

1. rank-1 B descriptor 被 driver 拒绝；
2. static barrier 让 dynamic shared base 偏移 16B，引发 misaligned address；
3. 3-stage 初始化遗漏第三个 barrier；
4. 用 Compute Sanitizer / racecheck 定位并修正。

### 正式结果

```text
group32: 最佳 TMA 比 same-run final 慢 3.32%
group64: 慢 0.12%
group128: 快 0.91%
```

group128 再做反序 200-sample A/B：

```text
差异仅 0.012%
```

判定持平。

### 为什么没有收益

此时虽然 copy issue 下降，但旧数据路径仍有：

```text
decoded FP16 B shared tile
CTA-wide epilogue shared scratch
约 32 KiB SMEM
```

因此 TMA 解决的不是当时最主要瓶颈。

### 结论

V15 不进入 final，但证明：

```text
SM120 TMA 可用
tensor map 正确
UTMALDG.2D 确实生成
```

### 下一步

真正要先解决的是 SMEM footprint。

### 源码 / 数据

```text
kernels/gemm_v15_tma.cu
results/tma_formal_4096_w20_i100.csv
results/tma_selected_pair_g128_w20_i200.csv
results/tma_*probe*.csv
```

---

## 18. V16a — Low-SMEM Warp Epilogue

### 当前缺陷 / 瓶颈

旧 final：

```text
32 KiB dynamic SMEM
94 registers/thread
Achieved occupancy 24.44%
```

其中很大一部分来自：

```text
64×128 FP32 CTA-wide epilogue scratch
```

### 怎么判断

NCU 已经显示：

```text
DRAM throughput 仅 3.46%
L2 hit 95.63%
```

说明继续优化 DRAM 没意义，而 occupancy / shared footprint 更值得处理。

### 解决方式

把 CTA-wide output scratch 改成：

```text
每个 warp 16×16 FP32 scratch
分块 half2 写回
```

### 同场性能

| Group | Old final |   V16a |       Improvement |
| ----: | --------: | -----: | ----------------: |
|    32 |    2.9181 | 2.6554 |  **+9.89%** |
|    64 |    3.2890 | 2.9098 | **+13.04%** |
|   128 |    3.2787 | 2.8718 | **+14.17%** |

### 资源变化

```text
SMEM 32 KiB → 24,064 B
Theoretical occupancy 25% → 33.33%
```

### 解决后还剩什么问题

虽然 epilogue 缩小了，但 B 仍然经过：

```text
packed INT4
→ decode 成完整 FP16 B shared tile
→ WMMA
```

decoded-B shared tile 成为下一大 footprint。

### 下一步

把 B decode 和 MMA 都搬进 register path。

### 源码 / 数据

```text
kernels/gemm_v16_epilogue.cu
results/v16a_formal_all_groups.csv
```

---

## 19. V16b — Lane-Native Repack + Register MMA

### 当前缺陷 / 瓶颈

V16a 仍有完整 decoded FP16 B shared tile。

这带来：

```text
INT4 decode 后 shared store
Tensor Core 前 shared load
scalar I2F/F2FP
地址计算
额外 SMEM
```

### 解决方式

重构数据路径：

```text
offline MMA-lane-native INT4 repack
→ packed B 进 SMEM
→ register 内 f16x2 bit-magic decode
→ × packed scale
→ explicit mma.sync.m16n8k16
→ FP32 accumulator
→ direct half2 global store
```

### INT4 decode

使用 bit trick：

```text
XOR sign bit
→ 嵌入 half 1024h
→ 减 1032h
→ half2 scale multiply
```

避免逐元素 int→float conversion。

### 同场性能

| Group |   V16a |   V16b |       Improvement |
| ----: | -----: | -----: | ----------------: |
|    32 | 2.7035 | 2.5650 |  **+5.40%** |
|    64 | 2.8853 | 2.5351 | **+13.81%** |
|   128 | 2.9288 | 2.5245 | **+16.01%** |

### 资源

```text
94 registers/thread
14,848 B SMEM
41.67% theoretical occupancy
0 spill
```

### 为什么有效

同时解决了两个问题：

```text
decoded-B SMEM footprint
shared store/load instruction
```

### 新瓶颈

此时 shared 已明显下降，原先 V15 中不重要的 copy issue 重新可能成为性能项。

### 下一步

重新测试 TMA。

### 源码 / 数据

```text
kernels/gemm_v16_register.cu
src/repack.cu
results/v16b_formal_all_groups.csv
```

---

## 20. V16c — TMA 在 Register-MMA 架构上重新测试

### 当前问题

V15 的 TMA 是在高-SMEM WMMA 路径上失败的。

V16b 已经把：

```text
SMEM 32 KiB → 14.8 KiB
```

且改成 register MMA，因此必须重新验证 TMA。

### 解决方式

测试：

```text
v16c0 2-stage no-swizzle
v16c1 2-stage swizzle
v16c2 swizzle + 3-stage
v16c3 + DMA warp
v16c4 no-swizzle 3-stage
v16c5 no-swizzle 4-stage
```

### 正式性能

| Group |   V16b |  V16c0 |      Improvement |
| ----: | -----: | -----: | ---------------: |
|    32 | 2.5179 | 2.4460 | **+2.94%** |
|    64 | 2.5334 | 2.4963 | **+1.49%** |
|   128 | 2.5252 | 2.5286 |           -0.13% |

### 各失败候选的缺陷

```text
swizzle:
register ≈126~128
→ occupancy / issue 代价过大

3/4 stage:
SMEM / barrier / live state 增加
→ 没有覆盖更多 latency

DMA warp:
160 threads + 额外控制
→ consumer 资源减少，反而更慢
```

### 结论

TMA 在 register-MMA 架构下终于有净收益，但只适合 g32/g64。

### 新问题

register path 改变后，旧的 warp-shape 结论可能失效。

因此 V16d 重新搜 warp tile。

### 源码 / 数据

```text
kernels/gemm_v16_tma.cu
results/v16c0_formal_all_groups.csv
results/v16_finalists_formal_all_groups.csv
```

---

## 21. V16d — Register Path Warp Retune

### 当前问题

V14 得出的“64×32 不如 32×64”结论来自旧 WMMA/shared-B 路径。

现在 V16b/c 已改成：

```text
lane-native B
register decode
explicit MMA
```

B reuse / A load / register balance 全部变化，因此必须重测。

### 解决方式

测试：

```text
16×64
32×32
64×32
```

### 结果

```text
16×64:
B decode 重复过多 → 明显慢

32×32:
4096³ 出现非确定稀疏数值误差 → 移出 registry

64×32:
性能非常接近甚至局部超过 32×64
```

正式 finalist：

| Group |   V16b |            V16c0 |     V16d2 64×32 |
| ----: | -----: | ---------------: | ---------------: |
|    32 | 2.5403 | **2.4845** |           2.4906 |
|    64 | 2.5351 | **2.4557** |           2.4867 |
|   128 | 2.5119 |           2.5449 | **2.4868** |

### 新发现

64×32 的核心优势是：

```text
同一个 B fragment 可服务更多 M16 MMA
```

这意味着可以降低：

```text
B decode / MMA 比例
```

但代价是 A fragment load 增加。

### 下一步

V17 专门解决 64×32 的 A-side load 成本。

---

## 22. Final V16 — Group-Aware Copy Path

最终按 group 选择：

```text
group32 → V16c0 TMA
group64 → V16c0 TMA
group128 → V16b cp.async
```

独立 formal run：

| Group |              Median |          TFLOP/s |
| ----: | ------------------: | ---------------: |
|    32 | **2.2228 ms** | **61.832** |
|    64 |           2.2387 ms |           61.392 |
|   128 |           2.2754 ms |           60.402 |

这是一轮非常快的独立 run，但由于 DVFS，不把它直接和 V17/V18 的另一次运行做代码收益比较。

### NCU：V11d → V16

```text
Tensor active       37.87% → 44.93%
Achieved occupancy  24.44% → 40.07%
Long Scoreboard      0.844 → 0.053
Short Scoreboard     1.870 → 0.234
SMEM                 32 KiB → 12.816 KiB
```

### 新暴露的缺陷

```text
Barrier    2.611
Math Pipe  7.492
```

此时已经可以明确：

```text
DRAM throughput ≈5%
L2 hit ≈97%
```

所以最终不再是 DRAM-bound。

新的瓶颈转成：

```text
INT4 decode
scale multiply
address / issue
TMA synchronization
```

### 下一步

V17 同时攻击：

```text
B decode reuse
A-side LDS
barrier
decode instruction
```

### 数据

```text
results/final_v16_e2e_formal_all_groups.csv
results/profile/summary.csv
```

---

## 23. V17a — `64×32` Warp Tile

### 当前缺陷

V16 的 32×64 warp tile 中，一个 B fragment 的 M 方向 reuse 仍有限。

### 解决思路

改成：

```text
64×32
```

让同一个 B fragment 服务更多 M16 MMA：

```text
B decode 次数 / MMA 次数 ↓
```

### 新问题

代价是：

```text
A fragment 数量 ↑
A-side shared load ↑
```

因此 V17a 本身并不是最终答案，而是为 V17b 暴露 A-load 问题。

### 数据

```text
results/v17_formal_all_groups.csv
results/profile/v17_summary.csv
```

---

## 24. V17b — `ldmatrix.x4`

### 当前缺陷 / 瓶颈

64×32 提高 B reuse 后，A-side load 变重。

原先 A fragment 使用多个 scalar LDS。

### 怎么判断

静态 SASS：

```text
V17a A-side LDS = 48
```

A load 指令数成为明显成本。

### 解决方式

改用：

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

将 Tensor Core 所需 A fragment 直接按硬件布局载入寄存器。

### 静态变化

```text
A-side LDS 48 → 24
```

### NCU

```text
V17a duration 2.6317 ms
V17b duration 2.6114 ms

Tensor active 45.89% → 46.25%
```

### 正式结果

```text
g32 2.3675 ms / 58.053 TFLOP/s
g64 2.4306 ms / 56.545 TFLOP/s
```

### 新问题

B decode / scale 本身仍然存在重复。

所以下一步做 fragment-pair。

### 源码 / 数据

```text
kernels/gemm_v17_instruction.cu
results/v17_formal_all_groups.csv
results/profile/v17_summary.csv
```

---

## 25. V17c — Fragment-Pair Repack

### 当前缺陷

当前两个 k16 fragment 仍分别 load / decode，scale broadcast 也存在重复。

### 解决方式

将两个 k16 fragment 合并：

```text
uint32 pair
low16  = k16 0
high16 = k16 1
```

同时把 scale broadcast 提到 k16 loop 外。

目标：

```text
B shared load ↓
B decode reuse ↑
scale load ↓
```

### 局部效果

LDS / decode 路径确实改善。

### 但出现新的严重缺陷

```text
96 regs → 98 regs
```

跨过 occupancy allocation cliff：

```text
5 CTA/SM → 4 CTA/SM
Achieved occupancy 39.87% → 32.29%
```

### 性能结果

group32 总吞吐下降。

但 group128 因 scale / pair reuse 更大，正式结果成为 winner：

```text
2.4444 ms
56.226 TFLOP/s
+3.88% vs same-run old final
```

### 结论

fragment-pair 思路本身有效，但 live range 太长。

### 下一步

先尝试解决同步，再尝试 decode 指令，但最终真正必须解决的是 register pressure。

---

## 26. V17d — Full/Empty mbarrier

### 当前缺陷

TMA stage 仍需要 CTA-wide synchronization。

Barrier 指标较明显。

### 解决方式

构建 stage ownership：

```text
producer issue
→ full barrier
→ warp leader wait
→ __syncwarp
→ consumer compute
→ empty barrier
→ producer reuse stage
```

### NCU

```text
Barrier 3.165 → 0.003
```

### 为什么还是没变快

因为 V17c/d 都是：

```text
98 registers
4 CTA/SM
~32% achieved occupancy
```

虽然 barrier 几乎清零，但 occupancy cliff 带来的损失更大。

### 结论

> 优化一个 profiler counter 不等于优化最终 latency。

### 下一步

继续验证 Math Pipe，再统一回头压寄存器。

---

## 27. V17e — BFE / PRMT Decode

### 当前缺陷

V16/V17 后新的主要 issue 是：

```text
packed f16x2 decode
scale
integer bit manipulation
```

Math Pipe 较高。

### 解决方式

用：

```text
BFE / PRMT
```

重新组合 nibble，减少部分 LOP3 / decode 依赖。

### NCU

```text
Math Pipe 7.278 → 6.410
Barrier ≈0.003
```

### 为什么仍未提速

虽然 Math Pipe 降低，但：

```text
PRMT 增加
98 registers
4 CTA/SM
```

导致整体吞吐仍不如 V17b。

### V17 的关键结论

三个方向都证明局部指标可改善：

```text
fragment-pair → LDS/reuse 好
full/empty    → Barrier 好
BFE/PRMT      → Math Pipe 好
```

但共同问题都是：

```text
register live range 太大
```

因此 V18 的目标非常明确：

> 不再堆新功能，先把 98/100 registers 压回 96，而且不能 spill。

### V17 数据

```text
results/v17_formal_all_groups.csv
results/ab_v17_g32_*.csv
results/ab_v17_g64_*.csv
results/ab_v17_g128_*.csv
results/profile/v17_summary.csv
```

A/B：

```text
g32 V17b +3.45%
g64 V17b +2.71%
g128 V17c +2.07%
```

---

## 28. V18a — Compact Pair：解决 Register Cliff

### 当前缺陷

V17c/d/e：

```text
98/100 registers
```

导致：

```text
5 CTA/SM → 4 CTA/SM
```

这是此时最明确的性能缺陷。

### 解决方式

压缩 live range：

```text
4 个 broadcast scale
→ 2 个 packed scale registers

只跨 k16 保留 2 个 uint32 B pair
另外 2 个 fragment 到使用点再 uint16 load

block 坐标不跨 mainloop 保活
```

### 结果

所有 group：

```text
96 registers
0 spill
5 blocks/SM
41.67% theoretical occupancy
```

### 性能

group32 顺序控制：

```text
V18a avg       2.5255 ms
V17 final avg  2.5472 ms

+0.86%
```

### 新问题

Barrier 仍然高，需要把 full/empty 的好处带回来，但不能重新增加 register。

---

## 29. V18b — Compact Pair + Full/Empty

### 当前缺陷

V18a 恢复 occupancy，但：

```text
Barrier ≈3.4
```

仍然明显。

### 解决方式

在 96-reg 预算内重新实现：

```text
computed parity
32-bit shared-address wait
full/empty mbarrier
token-free lane arrival
compile-time scale-row simplification
```

关键目标：

> **保留 V17d 的低 Barrier，但不再付出 98-reg 的代价。**

### 结果

```text
96 registers
0 spill
Barrier ≈0.003
```

group128 A/B：

```text
V18b avg       2.3563 ms
V17 final avg  2.4687 ms

+4.77%
```

因此：

```text
group128 final → V18b
```

### 新问题

Math Pipe 仍高，继续尝试 compact BFE/PRMT。

---

## 30. V18c — Compact BFE / PRMT

### 当前缺陷

Barrier 已几乎清零，但：

```text
Math Pipe ≈11
```

仍然是主要 stall。

### 解决方式

将 BFE/PRMT decode 改成更短 live range，只保留当前需要的一对 nibble temporary。

### NCU

```text
Math Pipe 11.353 → 9.835
Barrier ≈0.003
Registers = 96
Spill = 0
```

### 为什么最终仍未选择

尽管 profiler 的 Math Pipe 更好，但：

```text
Duration:
V18a 2.5996 ms
V18b 2.6036 ms
V18c 2.6179 ms
```

V18c 仍然最慢。

说明 PRMT/BFE 的额外 issue / dependency 抵消了 Math Pipe counter 的改善。

### 结论

最终版本仍根据**真实 latency**选择，而不是根据单个 profiler 指标。

---

## 31. 当前 Final — Group-Aware V18 / V17

最终 dispatch：

```text
group32  → V18a
group64  → V17b
group128 → V18b
```

为什么不是统一一个 kernel：

```text
g32:
compact pair + 96 regs 最平衡

g64:
V18a 与 V17b 基本持平，保留更稳定的 V17b

g128:
pair reuse + full/empty 的收益最大，V18b 明显胜出
```

共同资源：

```text
CTA tile       64×128×32
Warp tile      64×32
Threads/CTA    128
Stages         2
Registers      96/thread
Spill          0
Theoretical occupancy 41.67%
Achieved occupancy    ≈39.8%
```

最终 `30 warmup + 500 samples`：

| Group |        Final steady | Effective TFLOP/s | Separate dequant+cuBLAS |  Fused advantage |
| ----: | ------------------: | ----------------: | ----------------------: | ---------------: |
|    32 | **2.4035 ms** |  **57.183** |               2.4951 ms | **+3.81%** |
|    64 | **2.4128 ms** |  **56.962** |               2.4767 ms | **+2.65%** |
|   128 | **2.3770 ms** |  **57.820** |               2.5019 ms | **+5.25%** |

最终：

```text
max_abs_error = 0
local spill requests = 0
```

---

## 32. 每一阶段中间结果放置位置

### 版本源码

```text
kernels/
├── gemm_v00_reference.cu       # scalar baseline
├── gemm_v01_pairwise.cu        # pairwise INT4 / half2
├── gemm_v02_repack_simt.cu     # tile-contiguous repack
├── gemm_v03_tensorcore.cu      # first Tensor Core
├── gemm_v04_repack_tensorcore.cu
├── gemm_v09_async.cu           # cp.async
├── gemm_v15_tma.cu             # first TMA sweep
├── gemm_v16_epilogue.cu        # low-SMEM epilogue
├── gemm_v16_register.cu        # lane-native register MMA
├── gemm_v16_tma.cu             # TMA register-MMA
├── gemm_v17_instruction.cu     # V17 / V18
└── gemm_final.cu               # final group-aware dispatch
```

### V00 ~ V15 主时间线

```text
results/benchmark.csv
```

包含：

```text
V00 scalar
V01 pairwise
V02 repack
V03 Tensor Core
V04 CTA
V05 BK
V06 warp tile
V07 grid
V08 CTA alternative
V09 cp.async
V10~V12 padding
第一阶段 final
```

### V13 / V14 / V15 专项实验

```text
results/persistent_100.csv
results/warp_shapes_100.csv
results/tma_formal_4096_w20_i100.csv
results/tma_selected_pair_g128_w20_i200.csv
results/tma_*probe*.csv
```

### V16

```text
results/v16a_formal_all_groups.csv
results/v16b_formal_all_groups.csv
results/v16c0_formal_all_groups.csv
results/v16_finalists_formal_all_groups.csv
results/final_v16_e2e_formal_all_groups.csv
```

### V17

```text
results/v17_formal_all_groups.csv
results/ab_v17_g32_*.csv
results/ab_v17_g64_*.csv
results/ab_v17_g128_*.csv
```

### V18

```text
results/v18_reg96_formal_g32.csv
results/v18_all96_formal_g32.csv
results/v18_all96_formal_g64.csv
results/v18_all96_formal_g128.csv
results/ab_v18_g*_*.csv
results/ab_v18_all96_g*_*.csv
```

### 最终正式结果

```text
results/final_v18_all96_e2e_formal_all_groups.csv
```

### Nsight Compute

```text
results/profile/summary.csv
    # V00 / V03 / V09 / V11 / V16 的瓶颈迁移

results/profile/v17_summary.csv
    # V17a~V17e：Barrier / Math Pipe / Occupancy / Tensor active

results/profile/v18_all96_selected_summary.csv
    # 最终 V18 selected profile
```

### 完整逐版本记录

```text
OPTIMIZATION_LOG.md
```

其中每个版本都保留：

```text
修改前
→ 当前缺陷
→ 性能瓶颈
→ 假设
→ 修改
→ benchmark
→ profiler
→ 失败/成功原因
→ 下一步
```

推荐检查顺序：

```text
README 本节
    ↓
results/ 对应 CSV
    ↓
kernels/ 对应版本源码
    ↓
results/profile/ NCU summary
    ↓
OPTIMIZATION_LOG.md
```

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
make -j 

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
include/          encoding, layout, benchmark and launch interfaces
src/              quantize, repack, CPU reference and benchmark harness
kernels/          V00~V18 + final，各阶段独立优化版本源码
tools/            benchmark / autotune / NCU parsing
results/          V00~V18 中间 benchmark、A/B 对照、最终 formal CSV、hardware
results/profile/  V00/V03/V09/V11/V16/V17/V18 的 Nsight Compute 汇总
OPTIMIZATION_LOG.md
                  完整逐版本优化记录、Profiler 判断与失败实验
```
