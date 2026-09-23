# RTX 5070 INT4 GEMM optimization log

本日志只引用 RTX 5070 实机数据。V00--V15 的主时间线来自
`results/benchmark.csv`（20 warmup、100 samples）；V16 正式候选来自对应的
`results/v16*.csv`；V17 正式候选、交错 A/B 与最终验收分别来自
`results/v17_formal_all_groups.csv`、`results/ab_v17_*.csv`、
`results/final_v17_e2e_formal_all_groups.csv`；V18 寄存器周期使用
`results/v18_*.csv`、`results/ab_v18_g*_*.csv` 和
`results/final_v18_e2e_formal_all_groups.csv`（中间 dispatch）与
`results/final_v18_all96_e2e_formal_all_groups.csv`（最终 30 warmup、500 samples）。固定
4096³、CUDA Event、以 median 排序。版本百分比只用
同一次独占运行中的行，避免把跨运行 DVFS 波动误写成代码收益。

effective TFLOP/s 一律按 `2*M*N*K/time` 计算，不把 dequant 指令计入 FLOP 数。NCU
duration 因 replay/采集开销只用于同类报告对比，不能替代 CUDA Event benchmark。

## Version V00 — scalar fused baseline

### 修改前

只有 CPU FP32 reference、GPU quant/dequant、cuBLAS 和 benchmark 基础设施，没有 fused
kernel。

### 当前瓶颈

需要先建立最简单、可验证的 fused baseline，确认 packed nibble、sign extension 与 scale
语义，而不是从复杂 Tensor Core 路径开始。

### 修改

每个 thread 计算一个 C 元素，直接读取 row-packed INT4，逐项 unpack、乘 group scale、转
half 并累加。

### 为什么

它是后续版本共同的 correctness oracle 和速度基线，保留最透明的数据路径。

### 性能

After: 234.3499 ms，0.586 TFLOP/s。作为 fused baseline，speedup=1.00x。

### Profiling

38 registers/thread，0 dynamic SMEM，理论 occupancy 100%。NCU：270.821 ms、Tensor pipe
0%、achieved occupancy 99.82%、Math Pipe Throttle 4.71、Not Selected 6.04、零 spill。

### 结论

正确但完全受标量算术与指令吞吐限制；高 occupancy 不能弥补没有 Tensor Core 的问题。

### 下一步

先减少每个 INT4 pair 的地址/装载/存储指令，再引入 tile repack。

## Version V01 — pairwise INT4 load

### 修改前

v00 为每个 output 元素重复处理 nibble 和标量 store。

### 当前瓶颈

packed byte 内两个 INT4 没有一起消费，地址计算与输出事务冗余。

### 修改

一个 thread 消费一对 INT4，并用 `half2` 写回。

### 为什么

与物理 packing 对齐，减少 global load、unpack 和 store 指令。

### 性能

Before: 234.3499 ms / 0.586 TFLOP/s。After: 151.8928 ms / 0.905 TFLOP/s。
Speedup 1.543x，延迟降低 35.19%；相对 v00 1.543x。

### Profiling

40 registers/thread，0 SMEM，理论 occupancy 100%，零 spill。

### 结论

有效，但 SIMT 点积仍离目标两个数量级。

### 下一步

让 CTA 消费连续的 B tile，去除跨 K 行 gather。

## Version V02 — tile-contiguous repack with SIMT compute

### 修改前

v01 仍从逻辑 row-major packed B 读取，每个 K 行相隔 `N/2` bytes。

### 当前瓶颈

CTA 的 B 工作集不是连续段；mainloop 地址计算和访问局部性差。

### 修改

增加 GPU repack/inverse-repack 与独立 CPU mapping test。B 物理布局变为
`[N/128][K/32][32][128]` packed INT4，scale 变为
`[N/128][K/group][128]`；计算暂时仍是 SIMT。

### 为什么

每个 32x128 B tile 变成连续 2048 bytes，为 vector/cp.async 与 shared staging 建立基础。

### 性能

Before: 151.8928 ms / 0.905 TFLOP/s。After: 109.8615 ms / 1.251 TFLOP/s。
Speedup 1.383x，延迟降低 27.67%；相对 v00 2.133x。

### Profiling

48 registers/thread，0 SMEM，理论 occupancy 83.33%，零 spill。

### 结论

repack 本身有效，但真正的大幅提升必须来自 Tensor Core。

### 下一步

把 dequant 后的 FP16 tile 放进 shared memory 并由 WMMA 消费。

## Version V03 — first Tensor Core path

### 修改前

v02 使用 CUDA core 做标量/向量 FMA。

### 当前瓶颈

算术吞吐是绝对瓶颈，INT4 流量节省无法抵消 SIMT dot product。

### 修改

采用 64x128x32 CTA tile、32x32 warp tile、FP32 accumulator；B 在 mainloop 内 unpack /
scale / FP16 conversion 后进入 shared，使用 WMMA `m16n16k16`。

### 为什么

固定 4096³ 足以摊薄 staging，同一 dequantized B tile 可被多个 M 行复用。

### 性能

Before: 109.8615 ms / 1.251 TFLOP/s。After: 5.3245 ms / 25.812 TFLOP/s。
Speedup 20.633x，延迟降低 95.15%；相对 v00 44.01x。

### Profiling

62 registers/thread，32 KiB SMEM，理论 occupancy 50%。NCU：Tensor active 19.14%，Long
Scoreboard 9.14，Short Scoreboard 5.14，Barrier 1.77，MIO 2.67，零 spill。

### 结论

Tensor Core 是决定性优化，但同步 global/shared pipeline 暴露了长延迟和 barrier stalls。

### 下一步

让 Tensor Core 路径直接消费 tile-repacked B，并搜索 CTA 形状。

## Version V04 — repacked WMMA CTA shapes

### 修改前

v03 已使用 Tensor Core，但仍消费普通 row-packed B。

### 当前瓶颈

B tile 搬运需要跨行寻址；CTA shape 也未由数据决定。

### 修改

把 tile-contiguous B/scale 接入 WMMA，并实测 64x128、128x64、128x128 三种 CTA。

### 为什么

比较 N 方向连续性、M 方向复用与 occupancy，避免凭经验固定 tile。

### 性能

| Candidate | Median (ms) | TFLOP/s | Regs | SMEM | Occupancy |
|---|---:|---:|---:|---:|---:|
| v04a 64x128 | 5.2368 | 26.245 | 62 | 32 KiB | 50.00% |
| v04b 128x64 | **5.0402** | **27.269** | 72 | 32 KiB | 50.00% |
| v04c 128x128 | 8.1366 | 16.891 | 64 | 64 KiB | 33.33% |

最佳 v04b 相比 v03 1.056x，延迟降低 5.34%；相对 v00 46.50x。

### Profiling

v04a NCU：6.222 ms、Tensor active 19.37%、Long Scoreboard 9.16、Barrier 1.72、MIO
2.63、achieved occupancy 49.09%。仅 repack 尚未消除同步流水线 stalls。

### 结论

repack 带来小幅收益；128x128 的 64 KiB shared 降低 residency 并明显回退，因此拒绝。

### 下一步

尝试增大 BK 减少 barrier 次数，再搜索 warp tile。

## Version V05 — BK64/BK128 barrier reduction

### 修改前

每个 K=32 tile 都需要 staging 与同步。

### 当前瓶颈

barrier 次数较多，但简单增大 BK 可能增加寄存器/SMEM 压力。

### 修改

实测 BK64 与 BK128，并复用 vectorized B tile load。

### 为什么

用实测判断较少同步能否覆盖更大的 live state 成本。

### 性能

| Candidate | Median (ms) | TFLOP/s | Regs | SMEM | Occupancy |
|---|---:|---:|---:|---:|---:|
| v05a BK64 | 5.7374 | 23.955 | 70 | 32 KiB | 50.00% |
| v05b BK128 | 5.6732 | 24.226 | 92 | 48 KiB | 33.33% |

最佳 v05b 仍比 v04b 慢 12.56%。

### Profiling

BK128 把 registers 提到 92、SMEM 提到 48 KiB、occupancy 降到 33.33%；零 spill。

### 结论

没有真正 load/compute overlap 时，扩大 BK 只增长 live state，拒绝作为最终方向。

### 下一步

在 BK64 上增大每 warp 的 N tile，以减少线程和共享 load 冗余。

## Version V06 — 32x64 warp tile

### 修改前

v05 使用较窄 warp tile，线程数与 barrier 参与者更多。

### 当前瓶颈

同一个 A fragment 可服务更多 B fragments，现有 warp tile 没有充分利用该寄存器复用。

### 修改

warp tile 改为 32x64，CTA 使用 4 warps/128 threads。

### 为什么

每次装入 A fragment 后执行更多 MMA，提升算术密度。

### 性能

Before: v05b 5.6732 ms / 24.226 TFLOP/s。After: 4.7065 ms / 29.202 TFLOP/s。
Speedup 1.205x，延迟降低 17.04%；相对 v00 49.79x。

### Profiling

100 registers/thread，32 KiB SMEM，理论 occupancy 25%，零 spill。

### 结论

尽管 occupancy 降低，fragment 复用收益更大；说明此阶段不是单纯 occupancy-bound。

### 下一步

测试 grid swizzle/B reuse 是否能继续降低 B 流量。

## Version V07 — B-reuse grid mapping

### 修改前

标准 grid 以 N-major 映射 CTA。

### 当前瓶颈

理论上相邻 M tile 可复用同一个 B tile，但是否受 L2 容量限制未知。

### 修改

改变 block 到 M/N tile 的映射，使连续 CTA 更偏向复用 B。

### 为什么

若 B 带宽是瓶颈，调度局部性应减少 L2/DRAM 请求。

### 性能

Before: 4.7065 ms / 29.202 TFLOP/s。After: 4.7884 ms / 28.702 TFLOP/s。
Speedup 0.983x，回退 1.74%。

### Profiling

100 registers/thread，32 KiB SMEM，25% occupancy。8 MiB packed weights小于 48 MiB L2，
steady-state 已有很高 cache reuse。

### 结论

grid swizzle 没有收益，最终恢复标准 N-major grid。

### 下一步

测试 128x64 CTA 是否能以更多 M 复用改善整体吞吐。

## Version V08 — 128x64 CTA alternatives

### 修改前

v06 的 CTA 为 64x128；v07 grid 方向失败。

### 当前瓶颈

需要验证更多 M 复用能否抵消更窄 N tile 和更高寄存器压力。

### 修改

实测 128x64x32 与 128x64x64，使用 32x64 warp tile。曾尝试 half-physical BN64
vector mapping，但 correctness relative L2 约 3.08%，在 benchmark 前拒绝并改为正确的标量
tile load。

### 为什么

错误候选绝不进入性能排名；正确候选用于验证 CTA 轴向选择。

### 性能

| Candidate | Median (ms) | TFLOP/s | Regs | SMEM | Occupancy |
|---|---:|---:|---:|---:|---:|
| v08a BK32 | 7.5832 | 18.124 | 96 | 32 KiB | 25.00% |
| v08b BK64 | 5.1610 | 26.631 | 122 | 32 KiB | 25.00% |

最佳 v08b 仍比 v06 慢 9.66%，比前一 v07 慢 7.78%。

### Profiling

v08b 达 122 registers/thread，增加 M 复用未弥补更差的加载/调度效率；零 spill。

### 结论

保留 64x128 CTA。错误 vector mapping 被完整记录但不作为性能点。

### 下一步

回到 v06 形状，使用真正的 asynchronous double buffering。

## Version V09 — `cp.async` pipeline

### 修改前

最优同步路径 v06 在每个 K tile 上串行 global load、同步、compute。

### 当前瓶颈

v03/v04 NCU 的 Long Scoreboard、Barrier 与 MIO stalls 很高，说明 latency 没有被隐藏。

### 修改

用 `cp.async` 对 A 和 packed B 做 2/3/4-stage pipeline；INT4 unpack、scale 与 FP16 转换仍在
tile 到达 shared 后、MMA 前发生。

### 为什么

异步预取让下一 K tile 的 global transfer 与当前 tile 的 Tensor Core 工作重叠。

### 性能

| Candidate | Median (ms) | TFLOP/s | Regs | SMEM | Occupancy |
|---|---:|---:|---:|---:|---:|
| v09s2 | **4.0547** | 33.896 | 93 | 32 KiB | 25.00% |
| v09s3 | 4.0832 | 33.659 | 93 | 32 KiB | 25.00% |
| v09s4 | 4.0532 | 33.909 | 93 | 32 KiB | 25.00% |

s4 单次表中只比 s2 快 0.04%，不足以证明额外 stage 更优；选择更简单、跨 probe 更稳定的
s2。v09s2 相比 v06 1.161x，延迟降低 13.85%；相对 v00 57.80x。

### Profiling

v09s2 NCU：Tensor active 24.46%，Long Scoreboard 1.52，Barrier 0.54，MIO 0.41，
achieved occupancy 24.77%，零 spill。Short Scoreboard 上升到 6.61，成为主瓶颈。

### 结论

`cp.async` 明确解决 global dependency/barrier 问题；继续增加 stage 没有稳定收益。

### 下一步

针对 SMEM 到 WMMA operand register 的 Short Scoreboard 做 stride padding。

## Version V10 — first shared-stride padding

### 修改前

v09s2 的 Short Scoreboard 为 6.61，远高于 Long Scoreboard 1.52。

### 当前瓶颈

WMMA operand 从共享内存读取时存在 bank/依赖冲突，继续优化 global pipeline 价值有限。

### 修改

分别测试 A stride +8、B stride +8、A/B 同时 +8。

### 为什么

改变相邻 shared 行的 bank 映射，减少 operand load serialization。

### 性能

| Candidate | Median (ms) | TFLOP/s | Regs | SMEM | Occupancy |
|---|---:|---:|---:|---:|---:|
| v10a A+8 | 3.5548 | 38.662 | 93 | 32 KiB | 25.00% |
| v10b B+8 | 2.7869 | 49.316 | 94 | 32 KiB | 25.00% |
| v10c A+8/B+8 | **2.6754** | **51.372** | 94 | 32 KiB | 25.00% |

v10c 相比 v09s2 1.516x，延迟降低 34.02%；相对 v00 87.59x。

### Profiling

资源几乎不变：94 registers/thread、32 KiB、25% theoretical occupancy、零 spill。大幅收益
在资源相同条件下出现，支持 shared bank/operand path 假设。

### 结论

B padding 是主要收益，A+B 组合进一步提升。

### 下一步

在 A/B padding 邻域做系统 sweep，并用 `<1%` 增益作为停止条件。

## Version V11 — local padding sweep

### 修改前

v10c 已达 2.6754 ms，但 +8/+8 未必是最优 B stride。

### 当前瓶颈

需要降低剩余 Short Scoreboard，同时避免 padding 增大 SMEM allocation class。

### 修改

测试 A+8/B+16、A+16/B+8、A+16/B+16、A+8/B+24。

### 为什么

搜索不同 bank 周期，保持最终 dynamic allocation 仍为 32 KiB（output staging 为最大 union）。

### 性能

| Candidate | Median (ms) | TFLOP/s |
|---|---:|---:|
| v11a A+8/B+16 | 2.7041 | 50.826 |
| v11b A+16/B+8 | 2.7420 | 50.124 |
| v11c A+16/B+16 | 2.8039 | 49.016 |
| v11d A+8/B+24 | **2.6719** | **51.439** |

v11d 相比 v10c 1.0013x，仅降低 0.13%；相对 v00 87.71x。

### Profiling

v11d NCU：3.1878 ms、Tensor active 37.87%、Long Scoreboard 0.84、Short Scoreboard
1.87、Barrier 0.19、MIO 0.05、Math Pipe 1.71、94 registers/thread、32 KiB、achieved
occupancy 24.44%、零 spill。相比 v09s2 的 NCU duration 降低 35.31%。

### 结论

padding 假设被 profiler 直接验证；v11d 是选择的最终实现。

### 下一步

扩展少量 B padding 邻域，确认没有稳定 >=1% 的遗漏点。

## Version V12 — stopping sweep

### 修改前

v11d 已在噪声范围内接近 v10c，但仍需验证更宽 padding。

### 当前瓶颈

Short Scoreboard 已显著降低，进一步 padding 很可能只有噪声级变化或破坏 bank pattern。

### 修改

测试 A+8/B+32、A+8/B+40、A+16/B+24。

### 为什么

用邻域实测决定停止，而不是主观宣布收敛。

### 性能

| Candidate | Median (ms) | TFLOP/s | vs v11d latency |
|---|---:|---:|---:|
| v12a A+8/B+32 | 3.0231 | 45.463 | +13.14% |
| v12b A+8/B+40 | 2.6812 | 51.261 | +0.35% |
| v12c A+16/B+24 | 2.7541 | 49.903 | +3.08% |

### Profiling

所有候选仍为 94 registers/thread、32 KiB、25% theoretical occupancy、零 spill。

### 结论

没有候选比 v11d 快 >=1%，满足停止条件；继续盲扫 padding 没有证据支持。

### 下一步

固化 v11d 为 `final`，跑三组完整 benchmark、cold-ish、最终 NCU 与指令核查。

## Version V13P — persistent CTA scheduling

### 修改前

normal final 为 2048 个 output CTA，每个 CTA 只计算一个 64x128 tile。原规范要求显式比较
normal 与 persistent scheduling。

### 当前瓶颈

如果 launch/scheduling 或跨 CTA cache locality 是主要损耗，减少 CTA 数并让每个 CTA 循环
处理多个 output tile 可能有利；如果单 CTA 内同步和尾部 residency 更重要，则会回退。

### 修改

增加独立 `v13p` kernel 实例：固定 128 CTA 的一维 grid，每个 CTA 用 grid-stride 顺序处理
16 个 tile（4096³ 共 2048 tiles），每个 tile 之间增加 CTA barrier，保持同一 2-stage
`cp.async`、A+8/B+24、64x128x32 数据路径。小尺寸 correctness 会把 grid clamp 到 tile 数。

### 为什么

128 正好整除 2048，避免静态 grid-stride 的 tile-count imbalance；同时不引入全局 atomic
counter 或额外 timed reset kernel。

### 性能

来源 `results/persistent_100.csv`，normal 与 persistent 在同一次 20/100 运行内比较：

| Group | Persistent v13p | TFLOP/s | Same-run normal final | Persistent penalty |
|---:|---:|---:|---:|---:|
| 32 | 3.2660 ms | 42.082 | 2.6953 ms | +21.18% |
| 64 | 3.5834 ms | 38.354 | 3.0219 ms | +18.58% |
| 128 | 3.5650 ms | 38.552 | 3.0121 ms | +18.36% |

### Profiling

group32 为 94 registers/thread、32 KiB、25% theoretical occupancy、零 spill，与 normal 的
资源级别相同。benchmark 已出现 21% 明确回退，因此没有把它升级为额外 NCU 关键版本。

### 结论

FAILED EXPERIMENT。减少 grid 中 CTA 数没有解决最终的 shared/register dependency；tile 间
额外 barrier、较少的硬件 CTA 调度自由度以及长寿命 CTA 反而降低吞吐。最终保留 normal grid。

### 下一步

不继续 persistent 参数搜索，回到 v11d normal final，并完成交付审计。

## Version V14 — 64x32 / 64x64 warp-tile completion sweep

### 修改前

正式路径已测过 32x32 与最终 32x64，但原始任务还把 64x32、64x64 列为重要候选示例。

### 当前瓶颈

最终 kernel 的 eligible warps 仍低；需要验证用一个 warp 覆盖更多 M 行、增加 B fragment
复用，是否能胜过当前增加 A fragment 复用的 32x64。

### 修改

把 async Tensor Core 模板推广为 compile-time `WARP_M/WARP_N`，保持相同 64x128x32 CTA、
2-stage `cp.async` 与 A+8/B+24 padding，新增 v14a=64x32（4 warps）和 v14b=64x64
（2 warps）。

### 为什么

只改变 warp tile，其他数据路径一致，可以直接判断 A/B fragment 复用和寄存器/occupancy
之间的权衡。

### 性能

来源 `results/warp_shapes_100.csv`，三种形状同一次 20/100 运行：

| Group | v14a 64x32 | v14b 64x64 | same-run final 32x64 |
|---:|---:|---:|---:|
| 32 | 2.7743 ms / 49.541 TF | 4.0485 ms / 33.948 TF | **2.7002 ms / 50.899 TF** |
| 64 | 3.0285 ms / 45.383 TF | 4.6803 ms / 29.366 TF | **3.0205 ms / 45.503 TF** |
| 128 | 3.0182 ms / 45.536 TF | 4.6168 ms / 29.769 TF | **3.0121 ms / 45.629 TF** |

group32 下 v14a 比 final 慢 2.74%，v14b 慢 49.93%。

### Profiling

v14a group32: 93 registers/thread、32 KiB、25% theoretical occupancy、零 spill。v14b:
202 registers/thread、32 KiB、64 threads/CTA、12.5% occupancy、零 spill。回退幅度已经远大于
1%，无需追加关键阶段 NCU replay。

### 结论

64x32 的更多 B reuse 未抵消 A operand 路径变化；64x64 的 16 个 accumulator fragments/warp
造成极高寄存器压力并把 active warp 上限减半。最终 32x64 是四类搜索中的实测最优。

### 下一步

保留 32x64 final，完成最终 clean build、correctness 和 PTX/SASS 审计。

## Version Final — selected A+8/B+24 stage-2 kernel

### 修改前

v11d 是 padding sweep winner，但需要独立入口、三组数据、cold-ish 和最终验收证据。

### 当前瓶颈

最终 NCU 显示 24.44% achieved occupancy、Short Scoreboard 1.87、Math Pipe 1.71。32 KiB
shared 只允许 3 CTA/SM；94 registers/thread 也压缩设计空间。

### 修改

`launch_final` 固定委托 v11d：64x128x32 CTA、32x64 warp、4 warps、2-stage cp.async、
A stride 40、B stride 152、标准 N-major grid。另提供 FP32 output correctness 实例。

### 为什么

这是所有正确候选中跨正式运行与 profiler 证据最一致的实现；stage 3/4、grid swizzle、
persistent scheduling、64x32/64x64 warp tile、128x64/128x128 CTA 与更宽 padding 均无
稳定收益。

### 性能

| Group | Final steady median | TFLOP/s | Cold-ish median | Quant | Repack | One-shot total |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | **2.6825 ms** | **51.236** | 2.8180 ms | 0.9528 ms | 0.0558 ms | 3.6911 ms |
| 64 | 2.8729 ms | 47.839 | **2.8744 ms** | 0.5702 ms | 0.0540 ms | 3.4972 ms |
| 128 | 2.8741 ms | 47.819 | 3.0423 ms | 0.4124 ms | 0.0540 ms | **3.3406 ms** |

group=32 steady 最快。`v11d` 行为 2.6719 ms，同一 kernel 的 `final` 重复行为
2.6825 ms；正式结论采用 `final`，不选择性使用更快重复值。v00 -> final 为 87.36x。

### Profiling

group32: 94 registers/thread、32 KiB dynamic SMEM、25% theoretical occupancy、24.44%
achieved occupancy、Tensor active/SM throughput 37.87%、DRAM throughput 3.46%、L2 hit
95.63%、11.73 active warps/SM、每 scheduler 0.413 eligible / 0.34 issued warps/cycle、
零 local spill requests。PTX/SASS 确认
`cp.async`、WMMA 与 `HMMA.16816.F32`，且没有 `LDL/STL` spill traffic。

### 结论

最终 kernel 正确并满足 fused mainloop 要求，但没有击败 separate dequant + cuBLAS：group32
2.6825 ms 对 2.2175 ms，慢 20.97%。它避免了 32 MiB FP16 B materialization，并保留 8 MiB
INT4 weights；这是内存占用/数据流收益，不应误写成 latency 胜利。

### 下一步

当前参数调优已收敛。下一阶段若继续，应改变实现层级：尝试 sm_120 原生 MMA/更适合
`ldmatrix` 的 warp-repacked operand、减少 32 KiB epilogue/input union、或以 CUTLASS/CuTe
实现更细粒度 load/dequant/MMA overlap。任何新方向仍需用相同 100-sample 与 NCU gate 验证。

## Version 15 — SM120 Tensor Memory Accelerator sweep

### 修改前

final 使用 128 个线程协同发出 16-byte `cp.async`，RTX 5070 的 CC 12.0 虽有 TMA 单元，
但工程此前没有 tensor-map/mbarrier 实现和实测数据。

### 当前瓶颈

final 的 DRAM throughput 只有 3.46%、L2 hit 95.63%，所以 TMA 不太可能通过原始带宽获益；
可能的收益来自减少 copy issue 指令，并让一个线程提交整块 A/B transaction。风险是 TMA
descriptor/mbarrier 开销、compact A bank conflict 和更高寄存器数。

### 修改

新增 `kernels/gemm_v15_tma.cu`：

- A tensor map：FP16 rank-2，global dimensions `{K,M}`，box `{32,64}`；
- packed-B tensor map：把 tile-contiguous 2 KiB tile 表示为 256 个 `uint64_t`，rank-2 box
  `{256,1}`；
- thread 0 发出 `cp.async.bulk.tensor.2d`，`mbarrier.arrive.expect_tx` 记录 2048 或 6144
  transaction bytes，consumer 用 parity wait；
- mbarrier 放在动态 shared 尾部，避免 static shared 把 128-byte-aligned TMA data base 偏移
  16 bytes；
- v15a：2-stage TMA B + cp.async padded A；
- v15b：2-stage TMA A/B，compact A；
- v15c：3-stage TMA A/B；
- v15d：4-stage TMA A/B，最终以 compact B 保住 occupancy；
- v15e：3-stage TMA A/B，仅 CTA leader wait 后 `__syncthreads`。

初版 rank-1 B descriptor 被 driver 以 `CUDA_ERROR_INVALID_VALUE` 拒绝，改成 rank-2 tile map。
初版 static barrier 使 dynamic shared base 位于 `+16B` 并触发 misaligned address；Compute
Sanitizer 定位后改为动态 shared 尾置 barrier。三 stage 初始化遗漏第三个 barrier 的问题也由
memcheck 的 illegal barrier arrive 报告定位。最终 v15e racecheck 为 0 errors / 0 warnings。

### 正确性与资源

256³ 的 group 32/64/128、1024³ group128 均通过，v15 与 final 的 `max_abs=0`。
所有实例零 spill。group32 的 v15a/v15b/v15c/v15d/v15e 分别为
96/122/120/100/120 registers/thread；动态 shared 为 32784/32784/32792/32800/32792 B。

### 正式性能

来源：`results/tma_formal_4096_w20_i100.csv`，4096³，20 warmups + 100 samples。

| Group | v15a | v15b | v15c | v15d | v15e | same-run final | 最佳 TMA 结论 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 3.2041 | **2.9995** | 3.0373 | 4.0693 | 3.0367 | **2.9031** | 慢 3.32% |
| 64 | 3.4265 | 3.2867 | **3.2611** | 5.0142 | 3.3192 | **3.2572** | 慢 0.12% |
| 128 | 3.4093 | **3.2302** | 3.2762 | 5.0165 | 3.2941 | 3.2600 | 快 0.91% |

group128 的 v15b 在独立 repeat 中为 3.2292 ms，对同次 final 3.2813 ms；由于收益小于
跨运行 DVFS 波动，又做了正反顺序 200-sample A/B。最终同进程反序文件
`results/tma_selected_pair_g128_w20_i200.csv` 给出 v11d 3.24509 ms、TMA 3.24547 ms，
中位数差 0.012%，判为持平。四 stage 即使把 shared 压到 32800 B 并恢复 25% occupancy，
compact B 的 bank conflict 仍使其约 4.08 ms。leader-wait 没有减少 registers，且更慢。

### 指令证据

`results/tma.ptx` 包含 `cp.async.bulk.tensor.2d...mbarrier`、
`mbarrier.arrive.expect_tx` 和 `mbarrier.try_wait.parity`；`results/final.sass` 包含
SM120 `UTMALDG.2D` 与 `SYNCS.ARRIVE.TRANS64`。这证明数据确实经过 TMA 单元。

### 结论

TMA 在 RTX 5070 上可用且实现正确，但在这个 64x128x32、L2-resident、每 tile 仅 6 KiB
输入的 fused GEMM 中没有稳定 >=1% 的 median gain。默认 `launch_final` 恢复并保留 v11d
cp.async；v15a--v15e 与全部 CSV 保留为可复现实验。TMA 与 TMEM 是不同能力；SM120 没有
SM100/SM110 `tcgen05.alloc` 风格的可编程 TMEM 路径。

## Separate baseline component run

`results/baseline_components.csv` 使用相同 20/100 规范，专门补齐独立 dequant latency：

| Group | Dequant only | cuBLAS only | Combined event |
|---:|---:|---:|---:|
| 32 | 0.1387 ms | 2.0422 ms | 2.2418 ms |
| 64 | 0.1358 ms | 2.0523 ms | 2.2363 ms |
| 128 | 0.1324 ms | 2.0395 ms | 2.2297 ms |

独立 median 不可直接相加；combined event 才是同一计时区间中的真实总 latency。完整主运行
中 separate totals 为 2.2175/2.2360/2.2124 ms，轻微差别属于两次独占运行间的频率噪声。

## Rejected experiments summary

- 128x128 CTA：64 KiB shared，occupancy 降低，明显回退。
- BK64/BK128 without overlap：寄存器/SMEM 增长超过 barrier 节省。
- B-reuse grid：8 MiB weights 已能驻留 48 MiB L2，steady-state 无收益。
- 128x64 CTA：最高 122 registers/thread，整体更慢。
- BN64 half-physical vector mapping：relative L2 约 3.08%，correctness gate 拒绝。
- Async 3/4 stages：正式数据与 stage2 相差不超过约 0.7%，没有稳定优势。
- 更宽 padding：没有 >=1% 增益，按预设停止条件结束。
- Persistent 128-CTA scheduling：group32 比同次 normal final 慢 21.18%，拒绝。
- 64x32 / 64x64 warp tile：group32 比同次 final 分别慢 2.74% / 49.93%，拒绝。
- TMA v15：最佳正式点相对 same-run final 为 group32 -3.32%、group64 -0.12%、
  group128 +0.91%；反序 200-sample group128 对照只差 0.012%，无稳定 >=1% 收益。

---

以下 V16 实验发生在上述第一次“Final”之后；本节取代旧 final 选择，但保留旧结论作为
真实历史。所有 V16 benchmark 均在 RTX 5070 上串行独占执行。

## Version V16a — low-SMEM warp epilogue

### 修改前

旧 v11d 把整个 64x128 FP32 output tile 放入 shared memory，再协作转 FP16 写回；连同
A/B pipeline 共使用 32,768 B dynamic SMEM。

### 当前瓶颈

NCU 的 achieved occupancy 只有 24.44%，每 SM 11.73 active warps。32 KiB/CTA 的 shared
占用和 94 registers/thread 一起限制 latency hiding；DRAM throughput 仅 3.46%，不是带宽瓶颈。

### 修改

每个 warp 只保留一个 16x16 FP32 scratch，分块写回 half2；4 warps 共 4 KiB epilogue
scratch。A/B `cp.async` mainloop 暂不变。

### 为什么

先单独消除 CTA-wide epilogue storage，可以验证 occupancy 假设，又不同时改变 operand
layout 和 MMA 语义。

### 性能

来源 `results/v16a_formal_all_groups.csv`，20/200：

| Group | Before old final | After V16a | Improvement |
|---:|---:|---:|---:|
| 32 | 2.9181 ms / 47.099 TF | 2.6554 ms / 51.758 TF | +9.89% |
| 64 | 3.2890 ms / 41.782 TF | 2.9098 ms / 47.234 TF | +13.04% |
| 128 | 3.2787 ms / 41.919 TF | 2.8718 ms / 47.858 TF | +14.17% |

### Profiling

96 registers/thread，24,064 B dynamic SMEM，33.33% theoretical occupancy，零 ptxas spill。
三组 `max_abs=0`。本阶段未单独做 NCU replay，因为资源下降和正式增益已支持继续推进。

### 结论

有效。旧 epilogue 的 shared-memory footprint 确实压低 resident warps。

### 下一步

decoded FP16 B tile 仍占主要 shared 空间，且 WMMA fragment API 迫使 B 先落 SMEM；改成
lane-native B layout 与显式 register MMA。

## Version V16b — lane-native repack and register MMA

### 修改前

V16a 仍把 2 KiB packed B 展开成 8 KiB FP16 shared B tile，再由 WMMA/ldmatrix 读取；
每个 nibble 经标量 sign extension 和 conversion。

### 当前瓶颈

在线 B decode 需要大量 shared store/load、地址计算及标量 I2F/F2FP。即使 epilogue 变小，
24,064 B SMEM 仍只得到 33.33% theoretical occupancy。

### 修改

- 离线 `repack_mma_kernel` 把四个同 lane 的 `m16n8k16` B operand 打包为一个 uint16；
- register 中用 XOR/1024h/1032h 技巧生成两个 f16x2，并乘广播 scale；
- 显式 PTX `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`；
- 按公开 lane mapping 把 FP32 accumulator 直接转 half2 写 global；
- shared 只保留两 stage A、packed B 和 scale。

### 为什么

把运行时 gather/address conversion 转成一次性 preprocessing；既减少在线指令，也同时
消除 decoded-B tile 和 epilogue scratch。

### 性能

来源 `results/v16b_formal_all_groups.csv`，20/200：

| Group | Before V16a | After V16b | Improvement |
|---:|---:|---:|---:|
| 32 | 2.7035 ms | 2.5650 ms / 53.583 TF | +5.40% |
| 64 | 2.8853 ms | 2.5351 ms / 54.215 TF | +13.81% |
| 128 | 2.9288 ms | 2.5245 ms / 54.441 TF | +16.01% |

### Profiling

94 registers/thread，14,848 B SMEM，41.67% theoretical occupancy，zero spills。MMA repack
通过独立 CPU word mapping；三个 group 的输出均 `max_abs=0`。

### 结论

有效，且 group 越大收益越明显。V16b 首次将完整在线数据路径变为 packed-B SMEM ->
register decode -> register MMA -> direct global store。

### 下一步

对照 CUTLASS/DeepGEMM 的 TMA pipeline，用 tensor map 替换仍由 128 threads 发出的 copy
instructions；同时实测 swizzle、stage 深度和 DMA warp，而不是假设 TMA 必然更快。

## Version V16c — full TMA register-MMA sweep

### 修改前

V16b 每 K tile 由 CTA threads 发出 A/B/scale 的 16-byte `cp.async`，搬运正确但消耗 issue
slots。RTX 5070 已由 V15 证明支持真实 TMA。

### 当前瓶颈

假设：在已经压低 SMEM 后，单线程提交整 tile TMA 可能释放 consumer issue bandwidth。
风险是 tensor-map 参数、mbarrier、swizzle 地址和额外 stage 增加 registers/同步成本。

### 修改

新增 `gemm_v16_tma.cu`，三张 tensor map 分别搬 A 4096 B、fragment-packed B 2048 B、
scales 256 B。测试：

- v16c0：2-stage，无 swizzle；
- v16c1：2-stage，A64/B128/scale128 swizzle；
- v16c2：swizzle + 3-stage；
- v16c3：swizzle + 3-stage + 专用 DMA warp（160 threads）；
- v16c4：无 swizzle 3-stage；
- v16c5：无 swizzle 4-stage。

初版 swizzle selector 错误导致 v16c1 correctness failure；按 16-byte atom 和 128-byte row
selector 修正后，c0--c5 三个 group 全部 `max_abs=0`。

### 为什么

这组实验完整覆盖开源高性能 GEMM 常见的 TMA、swizzle、加深 pipeline 与 producer warp
策略，允许数据决定它们是否适合 SM120 的小 6.25 KiB/stage fused workload。

### 性能

正式 c0 对照来自 `results/v16c0_formal_all_groups.csv`：

| Group | V16b | V16c0 TMA | Improvement |
|---:|---:|---:|---:|
| 32 | 2.5179 ms | 2.4460 ms / 56.190 TF | +2.94% |
| 64 | 2.5334 ms | 2.4963 ms / 55.057 TF | +1.49% |
| 128 | 2.5252 ms | 2.5286 ms / 54.354 TF | -0.13% |

group32 smoke 同场中 c0/c1/c2/c3 为 2.4531/2.6531/2.6953/2.8785 ms；另一轮
c4/c5 为 2.6163/2.7037 ms。swizzle、3/4 stage 和 DMA warp 都回退。

### Profiling

c0 group32/64 为 96 registers、12,816 B SMEM、41.67% occupancy；group128 为 98 regs、
33.33% occupancy。swizzle 版本升到约 126--128 regs；3/4-stage SMEM 为 19,224/25,632 B。
所有版本 zero spills。PTX 有 `cp.async.bulk.tensor...mbarrier`，SASS 有 `UTMALDG.2D`。

### 结论

两 stage、无 swizzle 的 TMA 只在 group32/64 达到稳定收益；group128 与 cp.async 持平略慢。
复杂 pipeline 因 register/occupancy/sync 代价失败。

### 下一步

固定两 stage，重新搜索 warp tile，最后按 group 选择 copy path。

## Version V16d — warp-tile retune after register MMA

### 修改前

旧 warp-shape 结论基于 WMMA + decoded-B SMEM；V16 register path 改变了 B reuse、register
pressure 和 epilogue，因此需要重新验证。

### 当前瓶颈

32x64 每 warp 有 16 个 MMA accumulator fragment。64x32 增加 A reuse，16x64 降低单 warp
accumulator 数但增加 warp/CTA 和 B decode 次数；没有理论结论可代替实测。

### 修改

将 register kernel 模板化为 `WARP_M/WARP_N/STAGES`，实现 v16d0 32x32（256 threads）、
v16d1 16x64（256 threads）、v16d2 64x32（128 threads）。

### 为什么

隔离 warp shape 变量，检查 operand reuse 与 occupancy 的新平衡。

### 性能

group32 smoke：v16d1 2.7746 ms / 49.534 TF；v16d2 2.4602 ms / 55.866 TF。
正式 finalist `results/v16_finalists_formal_all_groups.csv`：

| Group | V16b | V16c0 | V16d2 |
|---:|---:|---:|---:|
| 32 | 2.5403 | **2.4845** | 2.4906 |
| 64 | 2.5351 | **2.4557** | 2.4867 |
| 128 | 2.5119 | 2.5449 | **2.4868** |

### Profiling

v16d1：70 regs、14,848 B、50% occupancy；v16d2：93 regs、14,848 B、41.67%；zero spills。
v16d0 在 4096³ retry 的 `max_abs=1.703125`（虽仍落在宽松门槛内，但其他候选为 0），
并在多轮出现非确定稀疏偏差，因此移出 registry。v16d2 曾在第一次正式
group64 gate 出现一次非确定 failure，之后重复为 0，但不作为可靠 final。

### 结论

16x64 因重复 B decode 明显变慢；64x32 有竞争力但可靠性不足。保留已多次稳定的 32x64。

### 下一步

按 group 在 V16c0 和 V16b 中选择，执行最后 20/200、NCU 与指令审计。

## Version Final V16 — group-aware TMA/cp.async selection

### 修改前

V16c0 在 group32/64 快于 V16b，但 group128 略慢；单一 copy path 不是三组共同最优。

### 当前瓶颈

候选选择已经收敛。需要验证 dispatch、真实 preprocessing end-to-end、steady/cold-ish、
三组 correctness、NCU 和 PTX/SASS，而不是继续在 <1% 参数噪声中搜索。

### 修改

`launch_final` 对 group32/64 调 v16c0 TMA，对 group128 调 v16b cp.async。补充独立
`launch_repack_scales_128`，使 `repack_mma` 计时包含 B+scale；增加同一 CUDA Event 内的
quantize + final repack + final GEMM one-shot 行。

### 为什么

使用每组经过正式对照且可靠的 winner；端到端联合 Event 避免错误地相加几个独立 median。

### 性能

来源 `results/final_v16_e2e_formal_all_groups.csv`，20/200：

| Group | Final median / mean / min / P95 | TFLOP/s | Cold-ish | Quant | Repack B+scale | End-to-end |
|---:|---|---:|---:|---:|---:|---:|
| 32 | 2.2228 / 2.3308 / 2.2182 / 2.5314 ms | 61.832 | 2.3890 | 0.9589 | 0.0415 | 3.5444 |
| 64 | 2.2387 / 2.2380 / 2.2090 / 2.2420 ms | 61.392 | 2.2292 | 0.5736 | 0.0395 | 2.9872 |
| 128 | 2.2754 / 2.3172 / 2.2413 / 2.5460 ms | 60.402 | 2.2651 | 0.4122 | 0.0395 | 2.7509 |

同次 separate dequant+cuBLAS 为 2.2542/2.2550/2.2399 ms：final 在 group32/64 快
1.41%/0.73%，group128 慢 1.56%。全部 final rows `max_abs=0`。

### Profiling

NCU old v11d g32 -> final TMA g32：duration 3.1878 -> 2.6874 ms，Tensor active
37.87% -> 44.93%，achieved occupancy 24.44% -> 40.07%，Long/Short Scoreboard
0.844/1.870 -> 0.053/0.234。Barrier 升到 2.611，Math Pipe 升到 7.492。final cp.async
g128 的 Tensor active 44.07%、occupancy 40.03%、Barrier 1.186、Math Pipe 7.523。
两者 DRAM 约 5%、L2 hit 约 97%、local spilling requests=0。

### 结论

最终实现正确，并在同次 group32/64 运行中小幅击败 separate baseline。主要增益来自
consumer-native layout、寄存器 decode/MMA/direct epilogue 和更高 occupancy，不是 DRAM
带宽。TMA 只对 group32/64 有净收益。

### 下一步

参数级调优停止。未来工作应减少 f16x2 decode/scale/address issue，或用 CUTLASS/CuTe
SM120 collective 做等 workload 对照。当前 Math Pipe Throttle 约 7.5 是首要目标；TMA
group32 的 mbarrier/CTA sync 是次要目标。SM100 `tcgen05` TMEM 不适用于 RTX 5070。

## Version V17a--V17e — 64x32, ldmatrix, pair fragments, full/empty TMA, decode SASS

### 修改前

V16 final 的 group32 TMA 路径使用 `32x64` warp tile、每个 A fragment 四条 scalar
shared loads，并在每个 K tile 前后各执行一次 CTA-wide `__syncthreads()`。静态 SASS
每 32 HMMA 有 32 HADD2、32 HMUL2；NCU Math Pipe Throttle 约 7.5，Barrier 约 2.61。

### 修改顺序

- v17a：TMA 保持两 stage/无 swizzle，warp tile 改为 `64x32`；
- v17b：A operand 改为 `ldmatrix.m8n8.x4`，四组 8 lanes 分别提供 TL/BL/TR/BR 地址；
- v17c：新增 2 KiB/tile 的 `[warp_n][n8][lane][k16-pair]` 布局，一条 uint32 load
  同时得到两个 k16 fragment，四个 scale half2 broadcast 提到 k16 loop 外；
- v17d：每 stage 增加 empty mbarrier。warp leader 等 full 后 `__syncwarp` handoff，
  每 warp leader 在消费结束后 arrive empty，producer 等四个 warp 后才能覆盖 stage；
- v17e：只改变 register decode，显式 BFE/PRMT 组装 half2 nibble 位置。

pair repack 在 `src/repack.cu` 中实现，并在 harness 中加入独立 CPU byte reference；v17c--e
自动选择 pair buffer，其他 V16/V17 版本继续使用原 MMA fragment buffer。

### CUDA 13 静态结果

完整二进制由 `/usr/local/cuda-13.0/bin/nvcc` 13.0.88、`-arch=sm_120` 生成；所有实例
spill store/load 为 0。group32 SASS：

| Version | Registers | LDS | HMMA | HADD2/HMUL2 | LOP3 | PRMT | BAR.SYNC |
|---|---:|---:|---:|---:|---:|---:|---:|
| v17a | 96 | 48 | 32 | 16/16 | 92 | 0 | 3 |
| v17b | 96 | 24 | 32 | 16/16 | 92 | 0 | 3 |
| v17c | 98 | 20 | 32 | 16/16 | 91 | 0 | 3 |
| v17d | 98 | 20 | 32 | 16/16 | 94 | 0 | 1 |
| v17e | 98 | 20 | 32 | 16/16 | 82 | 16 | 1 |

强制 `__launch_bounds__(128,5)` 虽把寄存器压到 96，却产生 56--108 B spill traffic，已撤销。
让全部 consumer lanes 独立轮询 full barrier 会把 group32 推到 119 regs，也已撤销；保留
elected-lane acquire + warp handoff。V17e 达到明确的 LOP3 降幅，但 PRMT 增加，不能只凭
静态计数判定更快。

`ldmatrix` 初版为 98 regs；将 A tile 的 generic pointer 预转换为 32-bit shared address，
并在所有 fragment load 间复用后，v17b group32 降到 96 regs、无 spill。pair/scale 跨
k16 保活使 v17c--e 仍为 98 regs。

### 正确性与正式性能

CUDA 13 构建后，256³ group32/64/128 的全部 registry kernel 均 PASS；v17a--e 与新
final 都是逐元素零误差。`results/v17_formal_all_groups.csv` 的同场 20/200 winner：

| Group | Winner | Median ms | TFLOP/s | 相对同场旧 final |
|---:|---|---:|---:|---:|
| 32 | v17b | 2.3675 | 58.053 | +4.71% |
| 64 | v17b | 2.4306 | 56.545 | +3.12% |
| 128 | v17c | 2.4444 | 56.226 | +3.88% |

为排除 registry 顺序/温度偏差，又按 candidate/final/final/candidate 做四次独立 20/200。
两端 candidate 均值相对两端 final 均值的收益为 group32 v17b +3.45%、group64 v17b
+2.71%、group128 v17c +2.07%，12 次运行全部 `error=0`。

### NCU 裁决

group32 full replay：

| Version | Duration ms | Tensor active | Math Pipe | Barrier | Occupancy |
|---|---:|---:|---:|---:|---:|
| v17a | 2.6317 | 45.89% | 9.921 | 3.466 | 39.90% |
| v17b | **2.6114** | **46.25%** | 11.190 | 3.430 | 39.87% |
| v17c | 2.6753 | 45.12% | 7.855 | 3.165 | 32.29% |
| v17d | 2.6790 | 45.05% | 7.278 | **0.003** | 32.24% |
| v17e | 2.6838 | 44.98% | **6.410** | **0.003** | 32.24% |

v17b 将 A-side 静态 LDS 48→24 并取得最快 replay。v17c 的 98-reg live range 触发
5→4 blocks/SM occupancy cliff；所以 v17d 虽消除 barrier stall、v17e 虽降低 Math Pipe，
都没有净提速。group128 v17c 的 NCU duration 2.6807 ms，旧 final 为 2.7445 ms，
快 2.38%，足以保留 pair layout。

### 最终选择

`launch_final` 已更新为 group32/64→v17b、group128→v17c；harness 对 group128 final
自动选择 pair-repacked B。新 final 的独立 20/200 为 2.3709/2.3810/2.4517 ms，
57.970/57.723/56.059 TFLOP/s，三组 `max_abs_error=0`。绝对值受本轮 DVFS 影响，版本
收益只采用上述同场与交错 A/B。

下一步若继续，优先把 v17c--e 从 98/100 regs 压回 96 且不 spill，恢复 5 blocks/SM；
再尝试把 v17d handoff 或 v17e decode 移植到 v17b 的 occupancy 档位。

## Version V18a--V18c — compact live ranges and restored 5 blocks/SM

### 假设

V17c--e 的 pair/scale、full/empty 和 decode 改动改善了 LDS、barrier 与 Math Pipe 指标，
但 group32 的 98 registers 跨过 allocation cliff，只能驻留 4 CTA/SM。目标是在不使用
会产生 spill 的 `launch_bounds` 前提下压回 96 registers，恢复 5 CTA/SM。

### 修改

- v18a：四个广播 scale 合并为两个 packed scale register；只把两个 `uint32` B pair
  跨 k16 保活，另外两个 fragment 在使用点以 `uint16` 读取；
- v18b：computed parity、32-bit shared-address wait、full/empty mbarrier；empty barrier
  使用 128 个 token-free lane arrival，避免 arrival token 和 lane-election live range；
- v18c：BFE/PRMT decode 每次只保留一对 nibble 临时值；
- block M/N 坐标只在 TMA issue 与 epilogue 现场重算，不跨 mainloop 保活；
- TMA scale-row 按 group 编译期化简：group32 为 `2*tile`，group64 为
  `tile & ~1`，group128 为 `(tile >> 1) & ~1`，消除 `groups` 地址状态。

曾尝试将 empty barrier 改为每 warp lane0 单次 arrival；无论是否加 `__syncwarp`，lane
选举都会把 group32 v18b/c 推回 98 registers，因此撤销。

### 静态资源与正确性

CUDA 13.0.88、`sm_120` 的 group32/64/128 v18a/v18b/v18c 九个实例均为
96 registers，ptxas spill store/load 为 0；128 threads 下恢复 5 blocks/SM、理论
occupancy 41.67%。256³ 三组的所有 registry kernel 均 PASS，V18 和 final 逐元素零误差。

### 正式性能与顺序控制

`results/v18_reg96_formal_g32.csv`（30/500）中 v18a/b/c 为
2.5785/2.5919/2.5842 ms，当前 V17 final 为 2.6240 ms，三者都因恢复 occupancy 而重新
具备竞争力。candidate/final/final/candidate 的两端平均结果：

| Group | V18a avg | V17 final avg | V18a 相对 final |
|---:|---:|---:|---:|
| 32 | 2.5255 ms | 2.5472 ms | **+0.86%** |
| 64 | 2.5904 ms | 2.5631 ms | -1.05% |
| 128 | 2.6093 ms | 2.6144 ms | +0.19% |

随后用 scale-row 编译期化简将 group64/128 也压到 96 registers。新的正式 30/500 sweep
由 group64 v18a、group128 v18b 领先当前 final 1.82%/2.23%；正反顺序 A/B 为：

| Group | Candidate | Candidate avg | V17 final avg | 相对 final |
|---:|---|---:|---:|---:|
| 64 | v18a | 2.3797 ms | 2.3797 ms | +0.001% |
| 128 | v18b | 2.3563 ms | 2.4687 ms | **+4.77%** |

所以 group64 保留 v17b，group128 更新为 v18b。

### NCU

| Metric | v18a | v18b | v18c |
|---|---:|---:|---:|
| Duration ms | **2.5996** | 2.6036 | 2.6179 |
| Tensor active | **46.43%** | 46.39% | 46.13% |
| Achieved occupancy | 39.88% | 39.82% | 39.84% |
| Barrier | 3.398 | **0.003** | **0.003** |
| Math Pipe | 11.452 | 11.353 | **9.835** |
| Registers / local spill requests | 96 / 0 | 96 / 0 | 96 / 0 |

full/empty 与 BFE/PRMT 的目标 profiler 指标都按预期改善，但没有形成净吞吐优势。最终
瓶颈已不再是 group32 occupancy cliff，而是 compact pair 路径的 decode/scale/issue。

### 最终验证

更新后的 `launch_final` 为 group32 v18a、group64 v17b、group128 v18b。
`results/final_v18_all96_e2e_formal_all_groups.csv` 使用 30 warmups + 500 samples，三组
final steady 为 2.4035/2.4128/2.3770 ms（57.183/56.962/57.820 TFLOP/s），全部
error=0。selected NCU 中 group32/group128 achieved occupancy 为 39.88%/39.82%，
registers 96/96、local spill requests 0/0。
绝对值受 DVFS 影响；版本选择只使用同场与交错 A/B。
