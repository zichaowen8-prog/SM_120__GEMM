# RTX 5070 W4A16 Fused-Dequant GEMM 优化总结报告

## 1. 项目概述

本项目面向 **NVIDIA GeForce RTX 5070（Blackwell，SM120 / Compute Capability 12.0）**，针对固定矩阵规模 `M=N=K=4096` 实现并逐步优化一个 **W4A16 Weight-Only Fused-Dequant GEMM**：

```text
C[M,N] = A_fp16[M,K] × dequant(B_int4[K,N], scale)

A           : FP16
B           : signed INT4，2 个 INT4 / byte
Scale       : FP16 group-wise scale
Accumulator : FP32
Output      : FP16
```

最终实现不提前 materialize 完整 FP16 B，而是让 B 始终以 packed INT4 保存，离线阶段只做 quantization 与 memory repack；在线 GEMM mainloop 中才完成 INT4 unpack、group scale、FP16 fragment 生成，并立即送入 Tensor Core。

最终数据流：

```text
packed INT4 B
      ↓
offline memory repack
      ↓
tile / warp / lane-native layout
      ↓
TMA / Shared Memory
      ↓
register INT4 unpack + FP16 scale
      ↓
mma.sync.m16n8k16
      ↓
FP32 accumulator
      ↓
direct half2 store
```

---

## 2. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| Architecture | Blackwell |
| Compute Capability | 12.0 / `sm_120` |
| SM 数量 | 48 |
| Warp Size | 32 |
| Registers / SM | 65536 × 32-bit |
| Opt-in Shared Memory / Block | 101376 B |
| Shared Memory / SM | 102400 B |
| L2 Cache | 48 MiB |
| Memory Bus | 192 bit |
| 理论显存带宽 | 672.048 GB/s |
| VRAM | 12 GB |
| Driver | 595.84 |
| CUDA Toolkit | 13.0 |
| nvcc | 13.0.88 |
| Nsight Compute | 2025.3.1 |
| 编译目标 | `-arch=sm_120` |

主要编译选项：

```bash
-O3
--use_fast_math
-lineinfo
-arch=sm_120
-Xptxas=-v,-warn-spills
```

最终关键 kernel 均检查 register spill，正式版本 **local spilling requests = 0**。

---

## 3. Benchmark 与正确性规则

### 3.1 性能计时

所有 kernel latency 使用 CUDA Event 测量。steady-state GEMM 不包含：

- allocation；
- quantization；
- preprocessing；
- memory repack。

最终正式结果使用：

```text
30 warmup
500 samples
```

版本选择主要看 median，同时保存 mean / min / P95。

### 3.2 Effective TFLOP/s

统一使用 dense GEMM 的逻辑 FLOP：

```text
2 × M × N × K
```

INT4 unpack、dequant 和 scale multiplication 不额外计入 FLOP，因此本文中的 TFLOP/s 是 **Effective Dense GEMM Throughput**，不是原生 INT4 Tensor Core TOPS。

### 3.3 正确性

正确性比较对象是：

```text
A × dequant(quantized B)
```

而不是未量化的原始 FP16 B。所有进入 final 的正式 kernel 都通过 correctness gate，最终正式结果均为：

```text
max_abs_error = 0
```

### 3.4 DVFS 处理

RTX 5070 同时承担桌面显示，不同 run 会受 GPU 频率和温度影响。因此：

- 同一次进程中的 A/B 对照用于判断代码收益；
- 不把不同 run 的最好 latency 直接相除并宣称为代码提升；
- 历史最低值只作为观察值，不作为版本裁决依据。

---

# 4. INT4 编码与 Memory Repack

## 4.1 原始 INT4

原始 B 使用 row-major signed INT4：

```text
logical = k * N + n

even logical → low nibble
odd  logical → high nibble

q = sign_extend_4bit(nibble)
q ∈ [-8, 7]

scale = scales[(k / GROUP_SIZE) * N + n]
B_dequant[k,n] = FP16(q * scale)
```

4096×4096 共 16,777,216 个权重。packed INT4 payload 仅为 8 MiB，而 FP16 B 为 32 MiB。

支持：

```text
GROUP_SIZE = 32 / 64 / 128
```

## 4.2 Tile-Contiguous Layout

V02 首先将 B 重排为：

```text
[N/128][K/32][32][128]
```

一个 CTA 使用的 `32×128` INT4 tile 物理连续，占 2048 B。这样后续可以使用连续 copy、cp.async 和 TMA，而不需要跨 K 行 gather。

## 4.3 Lane-Native MMA Layout

V16 进一步将一个 2 KiB B tile 排成 `mma.sync.m16n8k16` lane 的直接消费顺序：

```text
tile = nt*(K/32) + kt
word = [k16=2][warp_n=2][n8=8][lane=32]

n  = nt*128 + warp_n*64 + n8*8 + lane/4
k0 = kt*32 + k16*16 + (lane%4)*2
```

一个 `uint16` 中四个 nibble 对应：

```text
q[k0,   n]
q[k0+1, n]
q[k0+8, n]
q[k0+9, n]
```

正好是一条 `mma.sync.m16n8k16` 中某个 lane 所需的四个 B operand。

## 4.4 Fragment-Pair Layout

V17/V18 对 group32 / group128 进一步使用：

```text
word32 = [warp_n][n8][lane][k16 pair]
low16  = k16=0 fragment
high16 = k16=1 fragment
```

因此一次 `uint32` shared load 可同时得到两个 k16 fragment。Scale 也按 CTA tile 连续排布，尽量把运行时 permutation 和地址计算移到 preprocessing。

---

# 5. 优化全过程

## V00 — Scalar Fused Baseline

### 操作

每个 thread 计算一个 C 元素，直接读取 row-packed INT4，逐项 unpack、sign extension、乘 group scale 并累加。

### 性能

```text
234.3499 ms
0.586 TFLOP/s
```

### Profiling

```text
38 registers/thread
0 dynamic SMEM
Achieved occupancy ≈ 99.82%
Tensor active = 0
```

### 结论

正确但完全受标量算术限制。高 occupancy 不能弥补没有 Tensor Core。

---

## V01 — Pairwise INT4 Load + half2

### 操作

利用一个 byte 内两个 INT4，一次消费一对权重并使用 `half2` 写出。

### 性能

```text
234.3499 → 151.8928 ms
0.586 → 0.905 TFLOP/s
Speedup = 1.543×
Latency -35.19%
```

### 性能提升点

减少 global load、unpack、地址计算和 store 指令。

---

## V02 — Tile-Contiguous Memory Repack

### 操作

新增 GPU repack / inverse-repack 和 CPU mapping reference，将 B 改成 CTA tile-contiguous layout，scale 同步重排。

### 性能

```text
151.8928 → 109.8615 ms
0.905 → 1.251 TFLOP/s
Speedup = 1.383×
Latency -27.67%
```

### 性能提升点

把原来跨行、跨大步长的 B 访问变成 CTA 所需 2 KiB 连续 tile，为后续异步搬运打基础。

---

## V03 — 首次 Tensor Core 路径

### 操作

引入：

```text
CTA tile  = 64×128×32
Warp tile = 32×32
FP32 accumulator
WMMA m16n16k16
```

B 在 mainloop 内 unpack / scale / FP16 conversion 后进入 shared，再由 Tensor Core 消费。

### 性能

```text
109.8615 → 5.3245 ms
1.251 → 25.812 TFLOP/s
Speedup = 20.633×
Latency -95.15%
```

### NCU

```text
Tensor active       19.14%
Long Scoreboard      9.14
Short Scoreboard     5.14
Barrier              1.77
MIO                  2.67
```

### 结论

Tensor Core 是整个项目最大的单步提升。瓶颈从计算能力转移到 global/shared latency 与 operand dependency。

---

## V04 — CTA Shape Search

### 操作

测试：

```text
64×128
128×64
128×128
```

### 结果

| Candidate | Median | TFLOP/s | SMEM | Occupancy |
|---|---:|---:|---:|---:|
| 64×128 | 5.2368 ms | 26.245 | 32 KiB | 50% |
| 128×64 | **5.0402 ms** | **27.269** | 32 KiB | 50% |
| 128×128 | 8.1366 ms | 16.891 | 64 KiB | 33.33% |

### 结论

更大 CTA 并不必然更快。128×128 因 64 KiB shared 降低 residency 而回退。

---

## V05 — BK64 / BK128

### 操作

尝试通过增大 BK 减少 barrier 次数。

| BK | Median | TFLOP/s | Registers | SMEM | Occupancy |
|---:|---:|---:|---:|---:|---:|
| 64 | 5.7374 | 23.955 | 70 | 32 KiB | 50% |
| 128 | 5.6732 | 24.226 | 92 | 48 KiB | 33.33% |

### 结论

失败。没有 load/compute overlap 时，减少 barrier 的收益被 register live state、SMEM 和 occupancy 代价抵消。

---

## V06 — 32×64 Warp Tile

### 操作

Warp Tile 从 32×32 改为 32×64，让同一个 A fragment 服务更多 B fragment。

### 性能

```text
5.6732 → 4.7065 ms
24.226 → 29.202 TFLOP/s
Speedup = 1.205×
Latency -17.04%
```

### 资源

```text
100 registers/thread
32 KiB SMEM
25% theoretical occupancy
```

### 结论

即使 occupancy 降低，fragment reuse 的收益仍更大。

---

## V07 — B-Reuse Grid Mapping

### 操作

改变 CTA M/N tile 调度顺序，试图提高 B 的 L2 reuse。

### 性能

```text
4.7065 → 4.7884 ms
-1.74%
```

### 原因

packed B 仅 8 MiB，而 L2 为 48 MiB，steady-state 下 B 本身已经高度 cache-resident。

### 结论

失败，恢复标准 N-major grid。

---

## V08 — 128×64 CTA Alternatives

### 操作

测试 128×64 CTA 和不同 BK。

| Candidate | Median | TFLOP/s | Registers |
|---|---:|---:|---:|
| BK32 | 7.5832 | 18.124 | 96 |
| BK64 | 5.1610 | 26.631 | 122 |

一个 half-physical BN64 vector mapping 曾产生约 3.08% relative L2 error，在 benchmark 前由 correctness gate 拒绝。

### 结论

保留 64×128 CTA。

---

## V09 — cp.async 2/3/4-Stage Pipeline

### 操作

让 `load tile i+1` 与 `compute tile i` 重叠，测试 2/3/4 stage。

| Stage | Median | TFLOP/s |
|---:|---:|---:|
| 2 | **4.0547 ms** | 33.896 |
| 3 | 4.0832 ms | 33.659 |
| 4 | 4.0532 ms | 33.909 |

4-stage 优势只有约 0.04%，不稳定，因此选择更简单的 2-stage。

### NCU

```text
Long Scoreboard: 9.14 → 1.52
Barrier:         1.77 → 0.54
MIO:             2.67 → 0.41
Short Scoreboard          → 6.61
```

### 结论

Global memory latency 已经显著被隐藏，新的瓶颈变成 shared → operand register。

---

## V10 — Shared Memory Padding

### 操作

针对 Short Scoreboard 测试 shared stride padding。

| Layout | Median | TFLOP/s |
|---|---:|---:|
| A+8 | 3.5548 ms | 38.662 |
| B+8 | 2.7869 ms | 49.316 |
| A+8/B+8 | **2.6754 ms** | **51.372** |

### 性能提升

```text
4.0547 → 2.6754 ms
Speedup ≈ 1.516×
Latency -34.02%
```

### 结论

B shared-memory bank / operand path 是此前非常关键的隐藏瓶颈。

---

## V11 — Local Padding Sweep

### 操作

继续搜索 A/B stride：

```text
A+8/B+16
A+16/B+8
A+16/B+16
A+8/B+24
```

最佳：

```text
A+8/B+24
2.6719 ms
51.439 TFLOP/s
```

### NCU

```text
Tensor active       37.87%
Long Scoreboard      0.84
Short Scoreboard     1.87
Barrier              0.19
Math Pipe            1.71
Registers            94
SMEM                32 KiB
Achieved occupancy  24.44%
```

### 结论

padding 假设得到 profiler 直接验证。

---

## V12 — Stopping Sweep

### 操作

继续测试更宽 padding：

```text
A+8/B+32
A+8/B+40
A+16/B+24
```

没有任何候选获得稳定 >=1% 的收益。

### 结论

停止 padding 参数搜索。该阶段体现了“以实验停止，而不是凭感觉停止”的优化原则。

---

## V13 — Persistent CTA

### 操作

固定 128 个 persistent CTA，每个 CTA grid-stride 处理多个 output tiles。

### 结果

```text
group32: +21.18% latency
group64: +18.58%
group128:+18.36%
```

### 原因

额外 barrier、长寿命 CTA、减少硬件 scheduling 自由度，且 workload 并非 launch-bound。

### 结论

明确失败，不再继续 persistent 搜索。

---

## V14 — Warp Tile Completion Sweep

### 操作

在当时 WMMA/shared-B 架构下补测：

```text
64×32
64×64
```

### group32

```text
32×64 final : 2.7002 ms
64×32       : 2.7743 ms  (-2.74%)
64×64       : 4.0485 ms  (-49.93%)
```

64×64 使用 202 registers/thread，occupancy 仅 12.5%。

### 结论

在旧数据路径中 32×64 最优。后续 V16 register-MMA 改变数据流后，64×32 又被重新测试，并在 V17 成为有效方向，说明优化结论具有架构上下文。

---

# 6. 第一阶段 Final：V11d cp.async Kernel

当时的 Final：

```text
CTA        64×128×32
Warp       32×64
4 warps
2-stage cp.async
A stride   40
B stride   152
```

group32：

```text
2.6825 ms
51.236 TFLOP/s
```

但它仍慢于 separate dequant + cuBLAS。主要原因是 decoded FP16 B 和 CTA-wide FP32 epilogue scratch 仍占大量 shared memory，achieved occupancy 仅约 24.44%。因此继续改变实现层级。

---

# 7. V15 — SM120 TMA 初次探索

### 操作

新增真实 SM120 TMA：

```text
cuTensorMap
cp.async.bulk.tensor.2d
mbarrier
UTMALDG.2D
```

测试：

```text
v15a  TMA B + cp.async A
v15b  TMA A/B
v15c  3-stage
v15d  4-stage
v15e  leader-wait
```

### 调试记录

1. rank-1 B descriptor 被 driver 拒绝，改成 rank-2 tile map；
2. static barrier 让 dynamic shared base 偏移 16 B，导致 misaligned address；
3. 3-stage 初始化遗漏第三个 barrier，由 memcheck 定位；
4. 最终 racecheck 为 0 errors / 0 warnings。

### 正式结果

最佳 TMA 相对 same-run final：

```text
group32: -3.32%
group64: -0.12%
group128:+0.91%
```

group128 反序 200-sample A/B 最终只有 0.012% 差异，判定持平。

### 结论

TMA 本身不是自动加速器。在旧 WMMA + 大 shared 路径中，descriptor、mbarrier 和 register 成本抵消了 copy issue 节省。

---

# 8. V16 — 实现层级重构

## V16a — Low-SMEM Warp Epilogue

### 操作

旧路径把整个 64×128 FP32 output tile 放入 shared。V16a 改成每个 warp 仅保留一个 16×16 scratch，再分块 half2 写回。

### 同场性能

| Group | Before | V16a | Improvement |
|---:|---:|---:|---:|
| 32 | 2.9181 | 2.6554 | **+9.89%** |
| 64 | 3.2890 | 2.9098 | **+13.04%** |
| 128 | 3.2787 | 2.8718 | **+14.17%** |

### 资源变化

```text
SMEM: 32 KiB → 24,064 B
理论 occupancy: 25% → 33.33%
```

### 结论

CTA-wide epilogue shared footprint 是明确瓶颈。

---

## V16b — Lane-Native Repack + Register MMA

### 操作

彻底取消完整 decoded-B shared tile，改为：

```text
offline lane-native INT4 repack
→ register f16x2 decode
→ explicit mma.sync.m16n8k16
→ direct half2 epilogue
```

INT4 decode 使用：

```text
XOR sign bit
→ embed FP16 1024h
→ subtract 1032h
→ multiply half2 scale
```

避免标量 I2F/F2FP。

### 性能

| Group | V16a | V16b | Improvement |
|---:|---:|---:|---:|
| 32 | 2.7035 | 2.5650 | **+5.40%** |
| 64 | 2.8853 | 2.5351 | **+13.81%** |
| 128 | 2.9288 | 2.5245 | **+16.01%** |

### 资源

```text
94 registers/thread
14,848 B SMEM
41.67% theoretical occupancy
0 spill
```

### 结论

这是从“WMMA + decoded shared B”转向真正 fused register pipeline 的关键转折。

---

## V16c — Register-MMA + TMA

### 操作

在低-SMEM register path 上重新测试 TMA：

```text
v16c0  2-stage no-swizzle
v16c1  2-stage swizzle
v16c2  swizzle + 3-stage
v16c3  + DMA warp
v16c4  no-swizzle 3-stage
v16c5  no-swizzle 4-stage
```

### 正式收益

| Group | V16b | V16c0 | Improvement |
|---:|---:|---:|---:|
| 32 | 2.5179 | 2.4460 | **+2.94%** |
| 64 | 2.5334 | 2.4963 | **+1.49%** |
| 128 | 2.5252 | 2.5286 | -0.13% |

### 失败方向

- shared swizzle 将 register 推到约 126–128；
- 3/4 stage 增加 SMEM 与同步成本；
- 专用 DMA warp 增加到 160 threads；
- 都没有净收益。

### 结论

TMA 在 V15 的旧架构中无收益，但在 V16 register-MMA 架构中对 g32/g64 开始有收益。

---

## V16d — Register Path Warp Retune

重新测试：

```text
16×64
64×32
32×32
```

- 16×64 因重复 B decode 变慢；
- 64×32 有竞争力；
- 32×32 在 4096³ 出现非确定稀疏误差，被移除。

正式 finalist：

| Group | V16b | V16c0 | V16d2 64×32 |
|---:|---:|---:|---:|
| 32 | 2.5403 | **2.4845** | 2.4906 |
| 64 | 2.5351 | **2.4557** | 2.4867 |
| 128 | 2.5119 | 2.5449 | **2.4868** |

---

## Final V16 — Group-Aware Dispatch

```text
group32 → V16c0 TMA
group64 → V16c0 TMA
group128→ V16b cp.async
```

独立 formal run：

| Group | Median | TFLOP/s |
|---:|---:|---:|
| 32 | **2.2228 ms** | **61.832** |
| 64 | 2.2387 ms | 61.392 |
| 128 | 2.2754 ms | 60.402 |

这是历史上非常快的一次 run，但因 DVFS 不拿它和其他独立 run 做代码收益计算。

### NCU：V11d → V16

```text
Tensor active       37.87% → 44.93%
Achieved occupancy  24.44% → 40.07%
Long Scoreboard      0.844 → 0.053
Short Scoreboard     1.870 → 0.234
SMEM                  32 KiB → 12.816 KiB
```

新的瓶颈变成：

```text
Barrier    ≈ 2.6
Math Pipe  ≈ 7.5
```

即 INT4 decode、scale、instruction issue 和 TMA synchronization。

---

# 9. V17 — 针对新瓶颈继续优化

## V17a — 64×32 Warp Tile

动机：64×32 可以让一个 B fragment 被更多 M 方向 MMA 重用，从而减少 B decode / MMA 比例。

## V17b — `ldmatrix.x4` A Load

将多个 scalar shared load 改成：

```text
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

静态 A-side LDS：

```text
48 → 24
```

group32 NCU：

```text
v17a  2.6317 ms
v17b  2.6114 ms
Tensor active 45.89% → 46.25%
```

## V17c — Fragment-Pair Repack

一次 uint32 同时取得两个 k16 fragment，并将 scale broadcast 提到 k16 loop 外。

问题是：

```text
96 regs → 98 regs
5 CTA/SM → 4 CTA/SM
Achieved occupancy 39.87% → 32.29%
```

虽然 LDS/decode 指标改善，group32 总吞吐下降。

## V17d — Full/Empty mbarrier

加入 full/empty stage ownership，目标去掉 CTA-wide barrier。

Barrier：

```text
3.165 → 0.003
```

但仍受 98-reg occupancy cliff 影响，因此总 latency 没变快。

## V17e — BFE/PRMT Decode

进一步降低 LOP3，Math Pipe：

```text
7.278 → 6.410
```

但 PRMT 增加，且仍受 occupancy cliff 限制，总吞吐没有改善。

### V17 正式结果

| Group | Winner | Median | TFLOP/s | 相对旧 final |
|---:|---|---:|---:|---:|
| 32 | V17b | 2.3675 | 58.053 | +4.71% |
| 64 | V17b | 2.4306 | 56.545 | +3.12% |
| 128 | V17c | 2.4444 | 56.226 | +3.88% |

candidate/final/final/candidate 顺序控制后：

```text
group32 V17b +3.45%
group64 V17b +2.71%
group128 V17c +2.07%
```

---

# 10. V18 — 压回 96 Registers

V17 的主要问题是 98/100 registers 跨过 5→4 CTA/SM 的 allocation cliff。V18 的目标是保留 pair / barrier / decode 优化，同时在不 spill 的情况下回到 96 registers。

## V18a — Compact Pair

### 操作

- 四个广播 scale 合并成两个 packed scale register；
- 仅跨 k16 保留两个 `uint32` B pair；
- 其余 fragment 在使用点读取；
- block 坐标不再跨 mainloop 长期保活。

### 结果

```text
96 registers/thread
0 spill
5 blocks/SM
41.67% theoretical occupancy
```

## V18b — Compact Pair + Full/Empty

进一步加入：

- computed parity；
- 32-bit shared-address wait；
- full/empty mbarrier；
- token-free lane arrival；
- group-specific compile-time scale-row simplification。

对 group128：

```text
V18b avg       2.3563 ms
V17 final avg  2.4687 ms
Improvement    +4.77%
```

因此 group128 更新为 V18b。

## V18c — Compact BFE/PRMT

进一步减少 decode temporary。

NCU：

| Metric | V18a | V18b | V18c |
|---|---:|---:|---:|
| Duration ms | **2.5996** | 2.6036 | 2.6179 |
| Tensor active | **46.43%** | 46.39% | 46.13% |
| Occupancy | 39.88% | 39.82% | 39.84% |
| Barrier | 3.398 | **0.003** | **0.003** |
| Math Pipe | 11.452 | 11.353 | **9.835** |
| Registers | 96 | 96 | 96 |
| Spill | 0 | 0 | 0 |

V18c 的 Math Pipe 更低，但最终 latency 仍慢于 V18a/V18b，再次说明单一 profiler counter 不能代替总 latency。

---

# 11. 最终 Kernel

最终 dispatch：

```text
group32  → V18a
group64  → V17b
group128 → V18b
```

共同配置：

```text
CTA Tile       64×128×32
Warp Tile      64×32
Threads/CTA    128
Warps/CTA      4
Stages         2
Registers      96/thread
Accumulator    FP32
Output         FP16
```

Dynamic SMEM：

```text
group32 / 64 : 12,816 B
group128     : 12,832 B
```

理论 occupancy：

```text
41.67%
```

实测 achieved occupancy 约 39.8%，全部 final：

```text
local spill requests = 0
max_abs_error        = 0
```

---

# 12. 最终数据流

```text
FP16 A
  │
  └── TMA
        ↓
   2-stage A SMEM
        ↓
   ldmatrix.x4
        ↓
  A register fragment
        │
        ├───────────────────────────────┐
                                        ↓
Packed INT4 B                      mma.sync.m16n8k16
  │                                     ↓
  └── offline lane-native repack    FP32 accumulators
        ↓                               ↓
     TMA B                          direct half2 store
        ↓                               ↓
 packed B SMEM                         FP16 C
        ↓
 uint16 / uint32 per lane
        ↓
 f16x2 bit-magic decode
        ↓
 FP16 group scale
        │
        └──────────────────────────────→ MMA
```

---

# 13. 最终 4096³ 正式结果

来源：

```text
results/final_v18_all96_e2e_formal_all_groups.csv
```

条件：

```text
30 warmup
500 samples
无并发 GPU 进程
```

| Group | Quantize | Repack | One-shot E2E | Separate dequant+cuBLAS | Final steady | TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.0723 ms | 0.0334 ms | 3.7298 ms | 2.4951 ms | **2.4035 ms** | **57.183** |
| 64 | 0.5741 ms | 0.0396 ms | 3.1444 ms | 2.4767 ms | **2.4128 ms** | **56.962** |
| 128 | 0.4135 ms | 0.0314 ms | 2.8462 ms | 2.5019 ms | **2.3770 ms** | **57.820** |

同次运行中，相对 separate dequant + cuBLAS：

```text
group32: +3.81%
group64: +2.65%
group128:+5.25%
```

最终 fused-dequant 路径不仅省去了完整 FP16 B materialization，也获得了实际 latency 优势。

---

# 14. 整体性能演进

以 group32 的代表值观察：

| Version | 核心优化 | Median | TFLOP/s |
|---|---|---:|---:|
| V00 | scalar fused | 234.3499 ms | 0.586 |
| V01 | pairwise INT4 + half2 | 151.8928 ms | 0.905 |
| V02 | tile-contiguous repack | 109.8615 ms | 1.251 |
| V03 | Tensor Core | 5.3245 ms | 25.812 |
| V04 | CTA search | 5.0402 ms | 27.269 |
| V06 | 32×64 warp tile | 4.7065 ms | 29.202 |
| V09 | 2-stage cp.async | 4.0547 ms | 33.896 |
| V10 | shared padding | 2.6754 ms | 51.372 |
| V11 | padding winner | 2.6719 ms | 51.439 |
| V16a | low-SMEM epilogue | 2.6554 ms* | 51.758 |
| V16b | lane-native + register MMA | 2.5650 ms* | 53.583 |
| V16c0 | TMA register MMA | 2.4460 ms* | 56.190 |
| V17b | 64×32 + ldmatrix | 2.3675 ms* | 58.053 |
| Final V18 | compact 96-reg group-aware | 2.4035 ms** | 57.183 |

`*` 同场阶段数据。  
`**` 最新 30/500 独立 formal run。不同 run 受 DVFS 影响，不把绝对值直接当精确代码增益。

从 V00 的 0.586 TFLOP/s 到最终约 57–58 TFLOP/s，整体提升接近两个数量级。若只按绝对代表值估算，约为 **97×** 量级，但正式版本裁决仍以同场 A/B 数据为准。

---

# 15. Nsight Compute：瓶颈迁移

## V03

```text
Tensor active       19.14%
Long Scoreboard      9.14
Short Scoreboard     5.14
```

瓶颈主要是 global/shared latency。

## V09

```text
Long Scoreboard 9.14 → 1.52
Barrier         1.77 → 0.54
```

Global latency 得到明显隐藏，Short Scoreboard 上升为主要问题。

## V11

```text
Tensor active       37.87%
Short Scoreboard     1.87
Long Scoreboard      0.84
```

shared operand 路径显著改善，瓶颈转向 SMEM footprint 和 occupancy。

## V16

```text
Tensor active       44.93%
Achieved occupancy  40.07%
Long Scoreboard      0.053
Short Scoreboard     0.234
L2 hit              96.51%
DRAM throughput      4.74%
```

已经可以明确判断 final 不是 DRAM-bound。新瓶颈是：

```text
Barrier       ≈2.6
Math Pipe     ≈7.5
```

即 INT4 unpack、scale、地址/发射和 TMA synchronization。

## V18

最终 selected profile：

```text
Tensor active       ≈46.3%
Achieved occupancy  ≈39.8%
Registers           96
Spill               0
L2 hit              ≈96%
```

group128 V18b 的 Barrier 约 0.003，说明 full/empty handoff 成功消除 barrier stall。剩余最主要问题已转为 packed INT4 decode、scale multiply 和 instruction issue。

---

# 16. 失败实验与经验

## 16.1 128×128 CTA

64 KiB shared 导致 residency/occupancy 明显下降，说明更大 Tile 不等于更高性能。

## 16.2 BK64/BK128

在没有 overlap 时，register/SMEM 增长超过 barrier 减少的收益。

## 16.3 Grid Swizzle

8 MiB packed B 小于 48 MiB L2，steady-state 下 B 已高度 cache-resident，因此无收益。

## 16.4 128×64 CTA

最高约 122 registers/thread，整体更慢。

## 16.5 3/4-stage cp.async

没有稳定超过 2-stage；更深 pipeline 会增加 SMEM、register 和同步复杂度。

## 16.6 Persistent CTA

group32 延迟增加 21.18%，说明该 workload 不是 launch/scheduling bound。

## 16.7 64×64 Warp

202 registers/thread、12.5% occupancy，性能慢约 50%。

## 16.8 初代 V15 TMA

旧架构下无稳定收益，但 V16c 在低-SMEM register path 上取得 g32 +2.94%、g64 +1.49%，说明优化技术必须结合整体数据流判断。

## 16.9 `launch_bounds__(128,5)`

虽然可以压低寄存器数，却产生 56–108 B spill traffic，因此撤销。

## 16.10 V17d/V17e

Barrier 和 Math Pipe 指标分别明显改善，但 98 registers 触发 5→4 CTA/SM occupancy cliff，最终 latency 没有改善。

重要经验：

> 不能只优化单一 profiler counter，最终要看完整资源平衡和 end-to-end latency。

---

# 17. 最终性能提升点总结

最终高性能来自以下阶段共同作用：

1. **Packed INT4**：B 从 32 MiB FP16 降到 8 MiB INT4；
2. **Offline Memory Repack**：row-major → CTA-contiguous → lane-native → fragment-pair；
3. **Tensor Core**：V02→V03 从 1.251 提升到 25.812 TFLOP/s；
4. **cp.async Pipeline**：显著降低 Long Scoreboard；
5. **Shared Padding**：4.0547→2.6754 ms，解决 shared operand/bank 问题；
6. **Low-SMEM Epilogue**：去除 CTA-wide FP32 output scratch，提高 resident warps；
7. **Register Dequant + Register MMA**：消除 decoded FP16 B shared tile；
8. **TMA**：在低-SMEM register path 中减少 copy issue；
9. **ldmatrix.x4**：A-side LDS 从 48 降到 24；
10. **Register Live-Range Control**：98/100 regs 压回 96 regs，恢复 5 CTA/SM 且不 spill。

---

# 18. 剩余优化空间

目前已经可以基本排除：

```text
DRAM bandwidth
global-memory 主瓶颈
register spill
低 occupancy cliff
传统 Long Scoreboard
```

后续最值得继续优化：

### 18.1 INT4 Decode 指令数

重点减少：

```text
LOP3
HADD2
HMUL2
PRMT/BFE
```

尤其关注每 32 条 HMMA 对应多少 decode/scale 指令。

### 18.2 Compact Pair Issue Dependency

继续尝试更适合 half2 decode 的 nibble permutation 和更短 live range。

### 18.3 Full/Empty Handoff

group128 已证明 barrier 可以接近清零。若移植到 group32，必须保证：

```text
register 不增加
不 spill
不降低 CTA residency
```

### 18.4 CUTLASS/CuTe SM120 对照

后续最有意义的是使用相同：

```text
signed INT4
FP16 group scale
W4A16 fused-dequant workload
```

与成熟 SM120 collective 做公平对照，而不是与 NVFP4 原生低精度 Tensor Core 结果混用。

---

# 19. 最终结论

本项目从：

```text
234.3499 ms
0.586 TFLOP/s
```

的标量 fused INT4 GEMM 起步，经过：

```text
pairwise packing
→ tile-contiguous memory repack
→ Tensor Core
→ CTA / Warp Tile 搜索
→ cp.async
→ Shared Memory padding
→ low-SMEM epilogue
→ lane-native INT4 layout
→ explicit register MMA
→ TMA
→ 64×32 Warp Tile
→ ldmatrix.x4
→ fragment-pair layout
→ full/empty mbarrier
→ 96-register live-range tuning
```

最终形成 RTX 5070 SM120 专用的 group-aware W4A16 kernel：

```text
group32  → V18a
group64  → V17b
group128 → V18b
```

最新正式 30 warmup + 500 samples：

```text
g32 : 2.4035 ms / 57.183 TFLOP/s
g64 : 2.4128 ms / 56.962 TFLOP/s
g128: 2.3770 ms / 57.820 TFLOP/s
```

最终三组均实现：

```text
96 registers/thread
0 local spill
max_abs_error = 0
≈40% achieved occupancy
≈46% Tensor/SM active
```

同次运行相对 separate dequant + cuBLAS：

```text
+3.81%
+2.65%
+5.25%
```

整个优化过程展示了清晰的 bottleneck 迁移：

```text
SIMT compute
→ Tensor Core
→ Global latency
→ Shared operand dependency
→ Shared footprint / Occupancy
→ INT4 decode / Scale / Issue
```

最终 kernel 已经从 memory-bound / synchronization-heavy 实现演化为以 **INT4 decode、scale 和 instruction issue** 为主要限制的高性能 fused Tensor Core kernel，这也是继续优化时最值得投入的方向。
