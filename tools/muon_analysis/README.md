# Muon / LayerWise Newton-Schulz analysis tools

Standalone tools for measuring and optimizing `LayerWiseDistributedOptimizer` (Muon)
Newton-Schulz performance on real model configurations. Nothing here is imported by
training code, and none of it runs in CI.

The benchmark models one dist_muon optimizer step on Nemotron-3 Ultra at 128 GPUs, split
across the two axes a rank really pays: the dense weights over GTP=64, and the expert
weights over EGTP=2. The branch history is an optimization of that step from 1112.678 ms
to 137.467 ms, one accepted candidate per commit.

| file | what it answers | needs |
|---|---|---|
| `bench_ns_strategies.py` | Which Newton-Schulz distribution strategy is fastest for the shapes a rank actually owns? | GPUs, `torch`, `emerging_optimizers` |
| `bench_ns_egtp.sbatch` | Runs the benchmark over the EGTP axis on one node, NVLink disabled. | Slurm, pyxis |
| `bench_ns_gtp.sbatch` | Runs the benchmark over the GTP axis, 16 nodes for GTP=64. | Slurm, pyxis |
| `run_dist_muon_bench.sh` | **Entry point.** Submits both axes, waits, reports one combined `TOTAL_STEP_MS`. | Slurm |
| `kernels/fused_ns.py` | Fused fp32 prologue / bf16 epilogue around the batched NS chain. | `triton`, `emerging_optimizers` |
| `kernels/wire24.py` | 24-bit split transport codec (bf16 high half + 8 mantissa bits). | `triton` |
| `verification/` | The equivalence gates the accepted optimizations passed: 14 `check_*.py` + 13 `window_*.sbatch`. | Slurm, pyxis |

## Requirements

The Python side is self-contained -- `bench_ns_strategies.py` resolves `kernels/` from its
own directory -- but the container must provide `torch` (CUDA), `triton` and
`emerging_optimizers` (`newton_schulz`, `batched_tsyrk_ex`; asserted, not optional).

Every number below was measured in the **NT4 pretraining image**:

```
gitlab-master.nvidia.com/xren/nemo_megatron_perf_optimization:mcore-moe-pytorch26.07fix-temain4adad4c2-hybridep94a9f8f6-ncclmemfix-cutedslgdpv0.3.0-arm
```

| | |
|---|---|
| torch | `2.13.0a0+9186a08b2c.nvinternal.26.07.pin.mem.thresh` |
| CUDA | 13.3 |
| hardware | GB300, 4 GPUs/node, NV18 intra-node; 16 nodes for GTP=64 |

The torch build is an internal one, so the image is the reproducible unit -- not a pip
list. Point the drivers at it with `IMAGE_PATH`; they fail loudly rather than run without.

## Running it from a fresh clone

Nothing is hardcoded to a site, but three things are cluster-specific and must be supplied:

```bash
# 1. ROOT_DIR   -- the directory CONTAINING the checkout; runs/ is written under it.
#                 Defaults to the parent of wherever sbatch was invoked, which is right
#                 when you submit from the repo root. Otherwise pass it explicitly.
# 2. IMAGE_PATH -- the container described above.
# 3. partition / account -- override on the sbatch command line.

bash tools/muon_analysis/run_dist_muon_bench.sh          # both axes, one TOTAL_STEP_MS

# or one axis at a time, with the site settings spelled out:
sbatch -p <partition> --account=<account> \
       --export=ALL,USE_SYRK=1,ROOT_DIR=/path/containing/Megatron-LM,IMAGE_PATH=/path/img.sqsh \
       tools/muon_analysis/bench_ns_gtp.sbatch
```

`--segment=16` in the GTP driver is topology-specific; drop it on clusters without
segments. GTP degree and node count must agree: `nodes * GPUs_per_node == GTP`.

**The wrapper blocks** -- it submits both axes and polls until they finish (up to 24 h by
default, `MAX_WAIT_SECONDS`), because it has to read both logs to report one number. On a
busy partition run it detached. It cancels its jobs on SIGINT/SIGTERM.

### What you should see

```
[run_dist_muon_bench] gtp  fastest step: 46.158 ms (per_shape_fuse_sub_set)
[run_dist_muon_bench] egtp fastest step: 91.130 ms (per_shape_batch_sub_wire_set)
[run_dist_muon_bench] TOTAL_STEP_MS=137.288
```

That is a real run of this tree (jobs 3100651 / 3100652, 16:00 and 7:48) against the
137.467 ms recorded in the table below -- 0.13% apart, both axes selecting the same arms.
On different hardware the per-shape argmin will legitimately select differently and land
elsewhere; that is the benchmark working, not a regression.

## Benchmarking Newton-Schulz

The world size must equal the sharding degree being modelled, so the two axes need
different allocations:

```bash
sbatch tools/muon_analysis/bench_ns_egtp.sbatch                       # EGTP=2,  1 node
sbatch tools/muon_analysis/bench_ns_gtp.sbatch                        # GTP=64, 16 nodes
sbatch --nodes=2 --export=ALL,GTP=8 tools/muon_analysis/bench_ns_gtp.sbatch   # smaller trial
```

The EGTP wrapper disables NVLink, SHM and NVLS so the collectives take the scale-out
fabric, which is where an EGTP group sits in a real job. The GTP wrapper leaves NVLink
enabled, because a 64-rank GTP group fits inside one GB200/GB300 NVLink domain.

### Reading the output

`useful` FLOPs are the irreducible share: orthogonalizing the full matrix once, divided
across the group. `issued` FLOPs are what one GPU actually executes. Two consequences
worth keeping straight:

- `duplicated` recomputes the whole matrix on every rank, so its `useful TF/s` is already
  its `issued TF/s` divided by the group size. Do not discount it twice.
- `blockwise` is off by default. It orthogonalizes each block rather than the matrix, so
  it changes the update rather than distributing the same one, and it is not used in
  practice. Add it with `--modes blockwise duplicated distributed` if you want a floor;
  it issues *less* than useful, so its throughput is not comparable to the other two.

Every rank orthogonalizes concurrently and the step ends when the slowest finishes, so the
reported step cost is the max over rank profiles, not the mean.

## Measured on GB300 -- Nemotron-3 Ultra, 128 GPUs

`GTP=64 / EP=64 / EGTP=2`, 5 NS steps, `polar_express`, `--use-syrk`, bf16 GEMMs.
Newton-Schulz time per optimizer step, per GPU: the slowest rank profile on each axis, so
the total assumes a GPU drawing the worst profile on both.

**Every number in this table is `--use-syrk`**, the path this workload is locked to.

| | Phase-0 | now | speedup |
|---|---:|---:|---:|
| **GTP** (dense, 64-rank NVLink) | 622.051 | **46.342** | **13.4x** |
| **EGTP** (expert, 2-rank network) | 475.315 | **91.125** | **5.2x** |
| **total** | 1112.678 | **137.467** | **8.09x** |

Without `--use-syrk` the original baseline is **1390 ms**.

The accept commits on this branch are that trajectory, one optimization each, with the
measured delta, equivalence result and job ids in every message. The short version:

| what changed | Delta ms |
|---|---:|
| per-shape mode selection instead of one mode everywhere | -138 |
| **stop recomputing the same matrix on every rank** (subgroup duplication, owner-computes, batched experts) | **-619** |
| fuse and pipeline the exchanges (one a2a pair per region, async windows) | -117 |
| shave staging copies and wire bytes (bf16 output leg, self-block elision, 24-bit input leg) | -35 |
| push the duplication factor `g` to its floor, split inside the subgroup | -40 |

### Mode vocabulary

`duplicated_sub_g<N>` splits the group into `world/N` subgroups that each own a disjoint
set of matrices and duplicate `N` ways inside: `g = world` is `duplicated`, `g = 1` is
owner-computes. `_bal` pools the ownership deal across shapes by greedy LPT instead of
dealing each shape independently -- a shape with 3 matrices dealt over 16 subgroups
otherwise strands 13 of them and still costs one whole matrix. `_dist` additionally
row-splits a matrix inside its subgroup; `_pipeN` runs the exchange as `N` async windows;
`_wbf16out` / `_wbf16out24in` are the wire codecs (output leg bitwise, input leg lossy at
~2^-16 and gated on the workload's 1e-3 tolerance).

## Drift warning

`bench_ns_strategies.py` reimplements `LayerWiseDistributedOptimizer`'s per-buffer param
layout rather than importing it, so it will go stale if that changes. `owned_matrices`
mirrors `_emit_bucket`'s ordering, and its own docstring notes that bucketing is
deliberately ignored: that changes which matrices land together, not the set of shapes.

The model description is a 54-layer hybrid Mamba-MoE and is fixed -- "these constants
DEFINE the benchmark". Other models need those module constants updated; there are no
flags for them.
