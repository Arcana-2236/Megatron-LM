# Muon / LayerWise optimizer analysis tools

Standalone tools for reasoning about `LayerWiseDistributedOptimizer` (Muon) memory and
Newton-Schulz performance on real model configurations. Nothing here is imported by
training code, and none of it runs in CI.

| file | what it answers | needs |
|---|---|---|
| `padding_estimate.py` | How much of the optimizer's buffers is shard-imbalance padding, per buffer and per GPU count? | stdlib only |
| `bench_ns_strategies.py` | Which Newton-Schulz distribution strategy is fastest for the shapes a rank actually owns? | GPUs, `torch`, `emerging_optimizers` |
| `bench_ns_egtp.sbatch` | Runs the benchmark over the EGTP axis on one node, NVLink disabled. | Slurm, pyxis |
| `bench_ns_gtp.sbatch` | Runs the benchmark over the GTP axis, 16 nodes for GTP=64. | Slurm, pyxis |
| `run_dist_muon_bench.sh` | **Entry point.** Submits both axes, waits, reports one combined `TOTAL_STEP_MS`. | Slurm |
| `kernels/fused_ns.py` | Fused fp32 prologue / bf16 epilogue around the batched NS chain. | `triton`, `emerging_optimizers` |
| `kernels/wire24.py` | 24-bit split transport codec (bf16 high half + 8 mantissa bits). | `triton` |

## Why both

They answer the two halves of the same question. The layer-wise optimizer assigns whole
matrices to data-parallel shards, so a rank's cost depends on *which* matrices it drew.
`padding_estimate.py` models the memory consequence of that assignment; `bench_ns_strategies.py`
measures the time consequence. Both derive their shapes from the same model description
and the same compute-balanced assignment, so their rank profiles line up.

## Estimating padding

Pure arithmetic, so it runs anywhere:

```bash
python tools/muon_analysis/padding_estimate.py \
    --hybrid-layer-pattern 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME/*E/*E' \
    --hidden-size 2688 --ffn-hidden-size 3712 --moe-latent-size 672 \
    --num-experts 256 --moe-shared-expert-intermediate-size 3712 \
    --kv-channels 192 --num-attention-heads 32 --num-query-groups 1 \
    --mamba-num-heads 16 --mamba-num-groups 16 \
    --mtp-num-layers 2 --mtp-use-repeated-layer \
    --world-size 512 --expert-model-parallel-size 64 \
    --ddp-num-buckets 8 --total-params-per-rank 2846797792
```

Validated against a real 512-GPU run: bucket count and `dp_size` match the layout Megatron
logs, and padding agrees to within 0.2%.

## Requirements

The Python side is self-contained -- `bench_ns_strategies.py` resolves `kernels/` from its
own directory -- but the container must provide:

| | why |
|---|---|
| `torch` (CUDA), `triton` | the NS chain and both Triton kernel modules |
| `emerging_optimizers` | `newton_schulz`, `batched_tsyrk_ex`; asserted, not optional |

Point the drivers at it with `IMAGE_PATH`; they fail loudly rather than run without it.

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

Out of the box on the plain GEMM path the same configuration costs **~1.3 s** per step.
Everything below forces `--use-syrk`, the path this workload is locked to, which is where
the **1112.678 ms** starting point comes from.

| | Phase-0 | now | speedup |
|---|---:|---:|---:|
| **GTP** (dense, 64-rank NVLink) | 622.051 | **46.342** | **13.4x** |
| **EGTP** (expert, 2-rank network) | 475.315 | **91.125** | **5.2x** |
| **total** | 1112.678 | **137.467** | **8.09x** |

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

Both tools reimplement `LayerWiseDistributedOptimizer._compute_per_buffer_param_layout`
rather than importing it, which is what keeps `padding_estimate.py` dependency-free. They
will go stale if that function changes. Re-check against the reference numbers in
`padding_estimate.py`'s module docstring after touching the packer.

The model description in both files is a 54-layer hybrid Mamba-MoE and is fixed. Other
models need the constants updated; `padding_estimate.py` takes them as flags, while
`bench_ns_strategies.py` has them as module constants.
