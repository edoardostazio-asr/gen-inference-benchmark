"""Run matched request-response benchmarks against fal and MachGen.

    python -m bench.run --suite flux2-dev --n 10 --concurrency 1 --budget 1.00
    python -m bench.run --suite minimax-h3-768p-5s --n 4 --dry-run

Providers run side by side in the same time window, each limited to --concurrency in-flight jobs,
and job order is shuffled so neither provider systematically goes first. The planned spend is
checked against --budget (USD, all providers combined) before anything is submitted.
"""
import argparse
import asyncio
import json
import os
import random
import statistics
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

from bench.providers import MACHGEN_BASE, run_fal, run_fal_sync, run_machgen, run_machgen_sync
from bench.suites import PROMPTS, SUITES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))


def plan(suites, n, providers):
    jobs = []
    for s in suites:
        for i in range(n):
            # Same prompt and seed on every provider for job i, so outputs can be compared pairwise.
            prompt, seed = PROMPTS[i % len(PROMPTS)], 1000 + i
            for p in providers:
                jobs.append((p, s, i, prompt, seed))
    random.shuffle(jobs)
    return jobs


def planned_cost(jobs):
    per = {}
    for p, s, *_ in jobs:
        per[p] = per.get(p, 0) + SUITES[s]["est_cost"][p]
    return per


async def machgen_balance(client):
    r = await client.get(f"{MACHGEN_BASE}/billing/account",
                         headers={"Authorization": f"Bearer {os.environ['MACHGEN_API_KEY']}"})
    r.raise_for_status()
    return r.json()["balance_micros"] / 1e6


async def main(args):
    providers = args.providers.split(",")
    jobs = plan(args.suite, args.n, providers)
    cost = planned_cost(jobs)
    total = sum(cost.values())
    print(f"Plan: {len(jobs)} jobs  est. cost " + "  ".join(f"{p}=${c:.2f}" for p, c in cost.items())
          + f"  total=${total:.2f}  budget=${args.budget:.2f}")
    if total > args.budget:
        raise SystemExit("Planned spend exceeds --budget; nothing submitted.")

    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300)) as client:
        if "machgen" in providers:
            bal = await machgen_balance(client)
            print(f"MachGen balance ${bal:.2f}")
            if cost.get("machgen", 0) > bal:
                raise SystemExit("MachGen balance too low for this plan; top up or reduce --n.")
        if args.dry_run:
            return

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = os.path.join(ROOT, "outputs", run_id)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)
        log_path = os.path.join(ROOT, "results", "raw", f"{run_id}.jsonl")
        sems = {p: asyncio.Semaphore(args.concurrency) for p in providers}

        async def one(job):
            p, s, i, prompt, seed = job
            sync = args.sync and SUITES[s]["kind"] == "image"
            async with sems[p]:
                if p == "machgen":
                    fn = run_machgen_sync if sync else run_machgen
                    rec = await fn(client, s, SUITES[s], i, prompt, seed, out_dir)
                else:
                    fn = run_fal_sync if sync else run_fal
                    rec = await fn(client, s, SUITES[s], i, prompt, seed, out_dir, SUITES[s]["est_cost"]["fal"])
            rec.update(run_id=run_id, concurrency=args.concurrency, mode="sync" if sync else "queue")
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            e2e = f"{rec['t_download']:.2f}s" if rec["t_download"] else "-"
            print(f"  {p:8s} {s:22s} #{i:<3d} {rec['status']:12s} e2e={e2e} {rec['error'] or ''}"[:160])
            return rec

        recs = await asyncio.gather(*(one(j) for j in jobs))
    summarize(recs)
    print(f"\nRaw log: {log_path}\nOutputs: {out_dir}")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else float("nan")


def summarize(recs):
    print(f"\n{'provider':8s} {'suite':22s} {'ok/n':>6s} {'p50 e2e':>8s} {'p95 e2e':>8s} {'p50 srv':>8s} {'cost':>7s}  notes")
    keys = sorted({(r["provider"], r["suite"]) for r in recs})
    for p, s in keys:
        rs = [r for r in recs if r["provider"] == p and r["suite"] == s]
        ok = [r for r in rs if r["status"] == "COMPLETED" and r["t_download"]]
        e2e = [r["t_download"] for r in ok]
        srv = [r["server_gen_s"] for r in ok if r["server_gen_s"]]
        cost = sum(r["cost_usd"] or 0 for r in rs)
        notes = []
        if any(r.get("overflow_converted") for r in rs):
            notes.append(f"overflow={sum(bool(r.get('overflow_converted')) for r in rs)}")
        partners = sum(r.get("execution_provider") == "partner" for r in rs)
        if partners:
            notes.append(f"partner-run={partners}")
        print(f"{p:8s} {s:22s} {len(ok):>3d}/{len(rs):<2d} {pct(e2e, .5):8.2f} {pct(e2e, .95):8.2f} "
              f"{(statistics.median(srv) if srv else float('nan')):8.2f} {cost:7.3f}  {' '.join(notes)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", required=True, choices=sorted(SUITES))
    ap.add_argument("--n", type=int, default=1, help="jobs per provider per suite")
    ap.add_argument("--concurrency", type=int, default=1, help="max in-flight jobs per provider")
    ap.add_argument("--providers", default="machgen,fal")
    ap.add_argument("--budget", type=float, default=1.0, help="max planned USD spend for this run")
    ap.add_argument("--sync", action="store_true", help="image suites: use each provider's blocking endpoint")
    ap.add_argument("--dry-run", action="store_true")
    asyncio.run(main(ap.parse_args()))
