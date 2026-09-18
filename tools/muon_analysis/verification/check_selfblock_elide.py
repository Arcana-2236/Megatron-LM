# Probe + functional/timing window for the `egtp-selfblock-elide` candidate (stage 4,
# phase 1).
#
# spec.toml sets workload_mode = "inference", so the gate is EQUIVALENCE, not convergence.
#
# The candidate rests on ONE unestablished fact (stage-4/plan.md, "Notes/risk" (1)): with
# `shard_count = 2` on the egtp axis, HALF of every exchange payload is the rank's own
# block, and under `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` / `NCCL_NVLS_ENABLE=0` --
# the settings that make the 2-rank single-node proxy take the network path -- it is
# unknown whether that half traverses NET.
#
#   hypothesis A  NCCL services the self connection locally anyway  -> elision only
#                 replaces NCCL's own device copy with ours; expected delta ~0, and the
#                 candidate is WITHDRAWN (plan prices it at ~4.0 ms, under the 5.17 ms
#                 EGTP axis floor).
#   hypothesis B  the self block goes on the wire -> eliding it removes half the
#                 NCCL-resident union; plan prices it at -20.0 ms e2e.
#
# Three independent discriminators, all in the SAME 1-node/2-rank job so none costs its
# own slot:
#
#   (a) LOG EVIDENCE -- the sbatch driver runs with `NCCL_DEBUG=INFO` and
#       `NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH`, which prints the transport chosen per
#       connection at init. Read off what the self connection got. Parsed by the driver,
#       not here.
#   (b) ARITHMETIC A/B on RAW collectives, no log parsing needed: time a 2-rank
#       `all_to_all_single` carrying both blocks against a `batch_isend_irecv` carrying
#       the PEER block only, at the region's real per-chunk payload. A ratio near 1:1 says
#       the self block never hit the wire (A); near 2:1 says it did (B).
#   (c) HCA COUNT -- printed by the driver from the same NCCL_DEBUG=INFO output. plan.md
#       note (2): the proxy may bind fewer ConnectX-8 HCAs than the 2-node beta
#       measurement (job 2870326) did, which is a third, non-tunable explanation for the
#       achieved-beta shortfall and must be excluded before A or B is believed.
#
# Then the REAL-ARM A/B and the equivalence gate, on the accepted state
# (`newton_schulz_tp_subgroup(..., subgroup_size=1, batched=True, fused=True,
# pipelined=True)`, banked in stage-3 phase 3 as `per_shape_batch_sub_pipe_set`):
#
#   A) elided vs the ACCEPTED pipelined arm it substitutes -- must be BIT-IDENTICAL. The
#      own block is written from `stack` / into `result` instead of being round-tripped
#      through the collective; a local copy of the own block is bit-identical to receiving
#      it, so plan.md note (4) makes max_abs_diff = 0.0 a REQUIREMENT, not a tolerance.
#   B) elided vs the per-matrix `duplicated` loop -- the lineage mechanism every earlier
#      number on this axis was taken under.
#   C) elided matrix 0 vs the frozen Phase-0 reference tensor.
#   D) accepted pipelined arm vs the same duplicated loop -- the control that separates
#      "elision moved the wrong bytes" from "this arm was already off".
#   E) guards -- elision without `pipelined=True` must be a HARD ERROR, not a silent
#      no-op, because a silent no-op would let the arm be scored while running the code it
#      claims to replace.
#
# The timing half runs both arms at the REAL region count (192 matrices, 96 owned per
# rank) so the phase has a cheap read on direction and size before a scored job is spent.
# It is a window, not the scored number.

import argparse
import csv
import json
import os
import sys
import zlib

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "tools", "muon_analysis"))

from bench_ns_strategies import newton_schulz_tp_subgroup  # noqa: E402
from emerging_optimizers.orthogonalized_optimizers.muon_utils import (  # noqa: E402
    newton_schulz_tp,
)


def build_stack(tag, m, n, count, world, rank, seed):
    """Matrix 0 is capture_reference.py's frozen input; 1..n-1 are distinct random fill."""
    local_rows = m // world
    stack = torch.empty((count, local_rows, n), device="cuda", dtype=torch.float32)
    gen = torch.Generator(device="cuda")
    base = seed + zlib.crc32(tag.encode()) % 100000
    for j in range(count):
        gen.manual_seed(base if j == 0 else seed + 7919 * j + zlib.crc32(tag.encode()) % 100000)
        full = torch.randn((m, n), device="cuda", dtype=torch.float32, generator=gen)
        stack[j] = full[rank * local_rows : (rank + 1) * local_rows]
        del full
    torch.cuda.empty_cache()
    return stack


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def raw_transport_probe(group, world, rank, mb_per_block, warmup, iters, log):
    """Discriminator (b): does the self block cost wire time on a RAW collective?

    ``all_to_all_single`` moves ``world`` blocks per rank, of which exactly ONE is the
    rank's own. ``batch_isend_irecv`` over the real peers moves ``world - 1``. If the self
    block is local, the two cost the SAME (hypothesis A). If it is on the wire, the
    all_to_all costs ``world / (world - 1)`` times as much -- 2x at world = 2 (B).

    A third arm adds the local device copy the candidate actually pays in place of the
    self block, so the window prices the candidate and not just the collective.
    """
    elems = int(mb_per_block * (1 << 20) // 4)
    send = torch.empty(world * elems, device="cuda", dtype=torch.float32).normal_()
    recv = torch.empty(world * elems, device="cuda", dtype=torch.float32)
    peers = [s for s in range(world) if s != rank]
    gpeer = {s: torch.distributed.get_global_rank(group, s) for s in peers}

    def a2a():
        torch.distributed.all_to_all_single(recv, send, group=group)

    def peer_only():
        ops = []
        for s in peers:
            ops.append(torch.distributed.P2POp(
                torch.distributed.irecv, recv[s * elems : (s + 1) * elems], gpeer[s], group))
            ops.append(torch.distributed.P2POp(
                torch.distributed.isend, send[s * elems : (s + 1) * elems], gpeer[s], group))
        for w in torch.distributed.batch_isend_irecv(ops):
            w.wait()

    def peer_only_plus_copy():
        peer_only()
        recv[rank * elems : (rank + 1) * elems].copy_(send[rank * elems : (rank + 1) * elems])

    out = {
        "mb_per_block": mb_per_block,
        "world": world,
        "all_to_all_single_ms": bench(a2a, warmup, iters),
        "peer_only_p2p_ms": bench(peer_only, warmup, iters),
        "peer_only_plus_local_copy_ms": bench(peer_only_plus_copy, warmup, iters),
    }
    out["ratio_a2a_over_peer_only"] = out["all_to_all_single_ms"] / out["peer_only_p2p_ms"]
    # 1.0 => the self block was never on the wire; world/(world-1) => it was.
    out["ratio_if_self_on_wire"] = world / (world - 1)
    out["hypothesis"] = (
        "B" if out["ratio_a2a_over_peer_only"] > 1.0 + 0.5 * (out["ratio_if_self_on_wire"] - 1.0)
        else "A"
    )
    del send, recv
    torch.cuda.empty_cache()
    log("RAW_TRANSPORT_PROBE=" + json.dumps(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="equivalence.csv to write")
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--probe-out", default=None)
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--shapes", nargs="+", default=["5120x2048", "2048x5120"])
    ap.add_argument("--counts", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--timing-count", type=int, default=192)
    ap.add_argument("--batch-chunk", type=int, default=64)
    ap.add_argument("--pipe-chunks", nargs="+", type=int, default=[4])
    ap.add_argument("--probe-mb", nargs="+", type=float, default=[41.9, 251.7])
    ap.add_argument("--fused", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--coefficient-type", default="polar_express")
    ap.add_argument("--fp32-matmul-prec", default="medium")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    torch.set_float32_matmul_precision(args.fp32_matmul_prec)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    group = torch.distributed.distributed_c10d._get_default_group()
    world, rank = group.size(), group.rank()

    def log(message):
        if rank == 0:
            print(message, flush=True)

    def arm(stack, elide, pipelined=True, pipe_chunks=4, subgroup_size=1, batched=True):
        return newton_schulz_tp_subgroup(
            stack, steps=args.steps, coefficient_type=args.coefficient_type,
            tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
            subgroup_size=subgroup_size, batched=batched, batch_chunk=args.batch_chunk,
            fused=args.fused, pipelined=pipelined, pipe_chunks=pipe_chunks,
            elide_self=elide,
        )

    # ---- (b) raw arithmetic A/B, before anything allocates the big stacks --------------
    probes = [
        raw_transport_probe(group, world, rank, mb, args.warmup, max(args.iters, 10), log)
        for mb in args.probe_mb
    ]
    if rank == 0 and args.probe_out:
        with open(args.probe_out, "w") as fh:
            json.dump(probes, fh, indent=2)
            fh.write("\n")

    rows = []

    def record(shape, count, name, a, b, note=""):
        d = (a.float() - b.float()).abs()
        max_abs = d.max().item() if d.numel() else 0.0
        row = {
            "axis": "egtp", "shape": shape, "count": count, "arm": name, "world": world,
            "fused": int(args.fused),
            "max_abs_diff": f"{max_abs:.6e}",
            "mean_abs_diff": f"{(d.mean().item() if d.numel() else 0.0):.6e}",
            "bitwise_equal": bool(torch.equal(a, b)),
            "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if max_abs <= args.atol else "FAIL",
            "note": note,
        }
        rows.append(row)
        log(f"  {shape:>11} n={count} {name:<50} max_abs={row['max_abs_diff']} "
            f"bitwise={row['bitwise_equal']} {row['pass']}")

    # ---- equivalence -------------------------------------------------------------------
    for shape in args.shapes:
        m, n = (int(v) for v in shape.split("x"))
        assert m % world == 0, f"{shape}: rows must divide the {world}-rank group"
        local_rows = m // world
        tag = f"egtp_{m}x{n}"
        ref_path = os.path.join(args.reference_dir, f"{tag}.pt")
        reference = None
        if os.path.exists(ref_path):
            reference = torch.load(ref_path, map_location="cpu")[
                rank * local_rows : (rank + 1) * local_rows
            ].clone().cuda()
        else:
            log(f"  {shape}: no reference tensor at {ref_path}")

        for count in args.counts:
            stack = build_stack(tag, m, n, count, world, rank, args.seed)
            log(f"{shape}: count={count}, local {local_rows}x{n}, world={world}, "
                f"fused={args.fused}")
            duplicated = torch.stack([
                newton_schulz_tp(
                    stack[j], steps=args.steps, coefficient_type=args.coefficient_type,
                    tp_group=group, partition_dim=0, tp_mode="duplicated", use_syrk=True,
                )
                for j in range(count)
            ])
            for pc in args.pipe_chunks:
                accepted = arm(stack, elide=False, pipe_chunks=pc)
                elided = arm(stack, elide=True, pipe_chunks=pc)
                record(shape, count, f"A elided(pc={pc}) vs accepted pipelined arm",
                       elided, accepted, note="substituted arm; must be bitwise")
                record(shape, count, f"B elided(pc={pc}) vs duplicated-loop", elided,
                       duplicated, note="lineage mechanism")
                if reference is not None:
                    record(shape, count, f"C elided(pc={pc})[0] vs phase0-reference",
                           elided[0], reference, note="phase-0 lineage")
                record(shape, count, f"D accepted(pc={pc}) vs duplicated-loop", accepted,
                       duplicated, note="control")
                del elided, accepted
            del stack, duplicated
            torch.cuda.empty_cache()
        del reference
        torch.cuda.empty_cache()

    # ---- the precondition must be a HARD error, not a silent no-op ---------------------
    guard_shape = args.shapes[0]
    m, n = (int(v) for v in guard_shape.split("x"))
    guard = build_stack(f"egtp_{m}x{n}", m, n, 2 * world, world, rank, args.seed)
    guards = []
    for name, kwargs in (
        ("elide without pipelined", dict(elide=True, pipelined=False)),
        ("elide + subgroup_size > 1", dict(elide=True, subgroup_size=world)),
        ("elide + unbatched", dict(elide=True, batched=False)),
    ):
        try:
            arm(guard, **kwargs)
            guards.append((name, "NO-RAISE"))
        except (ValueError, RuntimeError) as exc:
            guards.append((name, f"raised {type(exc).__name__}"))
    del guard
    torch.cuda.empty_cache()
    log("precondition guards: " + ", ".join(f"{k}={v}" for k, v in guards))
    for name, outcome in guards:
        rows.append({
            "axis": "egtp", "shape": guard_shape, "count": 2 * world,
            "arm": f"E guard: {name}", "world": world, "fused": int(args.fused),
            "max_abs_diff": "0.000000e+00", "mean_abs_diff": "0.000000e+00",
            "bitwise_equal": True, "atol": args.atol, "rtol": args.rtol,
            "pass": "PASS" if outcome != "NO-RAISE" else "FAIL", "note": outcome,
        })

    # ---- timing window at the real region count ---------------------------------------
    timing = {}
    if args.timing_count:
        for shape in args.shapes:
            m, n = (int(v) for v in shape.split("x"))
            stack = build_stack(f"egtp_{m}x{n}", m, n, args.timing_count, world, rank,
                                args.seed)
            entry = {}
            for pc in args.pipe_chunks:
                entry[f"accepted_pc{pc}"] = bench(
                    lambda pc=pc: arm(stack, elide=False, pipe_chunks=pc),
                    args.warmup, args.iters)
                entry[f"elided_pc{pc}"] = bench(
                    lambda pc=pc: arm(stack, elide=True, pipe_chunks=pc),
                    args.warmup, args.iters)
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, entry)
            mx = {k: max(g[k] for g in gathered) for k in entry}
            timing[shape] = {
                "per_rank": gathered,
                "max": mx,
                "delta_ms": {f"pc{pc}": mx[f"elided_pc{pc}"] - mx[f"accepted_pc{pc}"]
                             for pc in args.pipe_chunks},
            }
            log(f"TIMING {shape} (max over ranks, count={args.timing_count}): "
                + ", ".join(f"{k}={v:.3f}ms" for k, v in mx.items())
                + " | delta " + ", ".join(f"{k}={v:+.3f}ms"
                                          for k, v in timing[shape]["delta_ms"].items()))
            del stack
            torch.cuda.empty_cache()

    if rank == 0:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "gate": "equivalence", "atol": args.atol, "rtol": args.rtol, "world": world,
            "counts": args.counts, "pipe_chunks": args.pipe_chunks, "fused": args.fused,
            "shapes": args.shapes,
            "all_pass": all(r["pass"] == "PASS" for r in rows),
            "all_bitwise": all(r["bitwise_equal"] for r in rows if r["arm"].startswith("A")),
            "rows": len(rows),
            "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows),
        }
        print("EQUIVALENCE_SUMMARY=" + json.dumps(summary), flush=True)
        if timing:
            print("TIMING_SUMMARY=" + json.dumps(timing), flush=True)
            if args.timing_out:
                with open(args.timing_out, "w") as fh:
                    json.dump(timing, fh, indent=2)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
