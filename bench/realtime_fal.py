"""Measure fal's real-time (WebSocket) image generation: connection setup and per-frame latency.

    python -m bench.realtime_fal --n 50 --size 512

fal's real-time path keeps one WebSocket open and bypasses its queue; it is offered only for
fast SDXL-class models (fast-lcm-diffusion, fast-turbo-diffusion). There is no fal real-time video.
Each request is sent only after the previous result arrives, so latency is a clean round trip.
"""
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone

import fal_client
from dotenv import load_dotenv

from bench.suites import PROMPTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
UNIT_PRICE = {"fal-ai/fast-lcm-diffusion": 0.000575, "fal-ai/fast-turbo-diffusion": 0.00125}  # $/compute-s


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def main(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    size = {512: "square", 1024: "square_hd"}[args.size]
    lat, errors, inference = [], [], []
    t0 = time.perf_counter()
    with fal_client.realtime(args.endpoint) as conn:
        t_open = time.perf_counter() - t0
        for i in range(args.n):
            t = time.perf_counter()
            # No `seed`: this endpoint's real-time path answers INTERNAL_ERROR when one is sent.
            conn.send({"prompt": PROMPTS[i % len(PROMPTS)], "image_size": size,
                       "num_inference_steps": args.steps, "sync_mode": True})
            try:
                res = conn.recv()
            except fal_client.client.RealtimeError as e:
                res = f"RealtimeError: {e}"
            dt = time.perf_counter() - t
            if isinstance(res, dict) and res.get("images"):
                lat.append(dt)
                if (res.get("timings") or {}).get("inference"):
                    inference.append(res["timings"]["inference"])
            else:
                errors.append(str(res)[:200])
    total_compute = sum(inference) if inference else sum(lat)
    summary = {
        "run_id": run_id, "endpoint": args.endpoint, "image_size": args.size, "steps": args.steps,
        "n": args.n, "ok": len(lat), "errors": errors[:5],
        "t_open_s": t_open,
        "first_request_s": lat[0] if lat else None,
        "latency_p50_ms": pct(lat[1:], .5) * 1000 if len(lat) > 1 else None,
        "latency_p95_ms": pct(lat[1:], .95) * 1000 if len(lat) > 1 else None,
        "latency_min_ms": min(lat[1:]) * 1000 if len(lat) > 1 else None,
        "server_inference_p50_ms": statistics.median(inference) * 1000 if inference else None,
        "frames_per_s_sequential": (len(lat) - 1) / sum(lat[1:]) if len(lat) > 1 else None,
        "est_cost_usd": total_compute * UNIT_PRICE.get(args.endpoint, 0),
    }
    os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "raw", f"fal_realtime_{run_id}.json"), "w") as f:
        json.dump({**summary, "latency_s": lat, "inference_s": inference}, f, indent=1)
    for k, v in summary.items():
        print(f"{k:26s} {round(v, 4) if isinstance(v, float) else v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="fal-ai/fast-lcm-diffusion", choices=sorted(UNIT_PRICE))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--size", type=int, default=512, choices=[512, 1024])
    ap.add_argument("--steps", type=int, default=4)
    main(ap.parse_args())
