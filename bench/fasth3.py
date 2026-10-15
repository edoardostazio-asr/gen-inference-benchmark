"""Reactor FastH3 (streamed MiniMax H3) measured on the same prompts and seeds as MachGen's MiniMax H3.

    python -m bench.fasth3 --clips 6 --seconds 5

All clips are enqueued at once with autoplay on, the way an app would line up scenes. Per clip we
time enqueue -> built (`clip_generated`) -> first frame on the track (`clip_started`) -> played out.
FastH3 builds clips one after another, so per-clip build time is the gap since the previous build
finished. Real-time factor = seconds of video built / wall-clock seconds spent building (>1 means
generation outruns playback). The session token caps the session length so a crash cannot bill on.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from PIL import Image
from reactor_sdk import Reactor

from bench.realtime_reactor import mint_jwt, pct
from bench.suites import PROMPTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
MODEL = "reactor/fast-h3"
RATE_PER_S = 0.0350  # USD per second of `ready` session time (api.reactor.inc/pricing, 2026-10-12)


async def main(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.model.split("/")[-1].replace("-", "")[:16]
    out_dir = os.path.join(ROOT, "outputs", f"{tag}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    reactor = Reactor(model_name=args.model, jwt=mint_jwt(args.model, args.max_session))

    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0  # noqa: E731
    status_log, messages, frame_times = [], [], []
    ready, all_done = asyncio.Event(), asyncio.Event()
    state = {}
    clips = []            # index = our job number i
    playing = {"i": None, "frames": 0}

    def on_status(s):
        status_log.append((round(now(), 3), str(s)))
        if str(s) == "ready":
            ready.set()

    def on_message(m):
        t = now()
        kind = m.get("type") if isinstance(m, dict) else None
        data = (m.get("data") or {}) if isinstance(m, dict) else {}
        if kind not in ("state_update", "queue_update"):
            messages.append((round(t, 3), kind, data))
        if kind == "state_update":
            state.update(data)
        clip = data.get("clip") or {}
        meta = clip.get("metadata")
        if meta is None or not str(meta).startswith("job:"):
            return
        c = clips[int(str(meta)[4:])]
        if kind == "clip_queued":
            c.update(t_queued=t, frames_planned=clip.get("frames"), seconds=clip.get("seconds"))
        elif kind == "clip_generated":
            c["t_generated"] = t
        elif kind == "clip_started":
            c["t_started"] = t
            playing.update(i=c["i"], frames=0)
        elif kind in ("clip_finished", "clip_stopped"):
            c.update(t_finished=t, end=kind)
        elif kind == "clip_failed":
            c.update(t_failed=t, error=str(data)[:300])
        if clips and all(cl.get("t_finished") or cl.get("t_failed") for cl in clips):
            all_done.set()

    reactor.on("status_changed", on_status)
    reactor.on("message", on_message)
    t_ready = t_disc = None
    try:
        await reactor.connect()
        await asyncio.wait_for(ready.wait(), timeout=300)
        t_ready = now()
        video = reactor.track("main_video")

        @video.on_frame
        def _frame(rgb):
            frame_times.append(now())
            i = playing["i"]
            if i is None:
                return
            playing["frames"] += 1
            clips[i]["frames_seen"] = playing["frames"]
            if playing["frames"] == int(args.seconds * 12):  # keep the middle frame for a visual check
                Image.fromarray(rgb).save(os.path.join(out_dir, f"clip_{i}.jpg"), quality=92)

        await reactor.send_command("set_clip_seconds", {"seconds": args.seconds})
        await reactor.send_command("set_autoplay", {"enabled": True})
        for i in range(args.clips):
            clips.append({"i": i, "prompt": PROMPTS[i % len(PROMPTS)], "seed": 1000 + i})
        for c in clips:
            c["t_enqueue"] = now()
            await reactor.send_command("enqueue", {"prompt": c["prompt"], "seed": c["seed"],
                                                   "seconds": args.seconds, "metadata": f"job:{c['i']}"})
        try:
            await asyncio.wait_for(all_done.wait(), timeout=args.max_session - 30)
        except asyncio.TimeoutError:
            print("timed out waiting for every clip to play out")
    finally:
        await reactor.disconnect()  # non-recoverable: releases the GPU and stops billing
        t_disc = now()

    # --- metrics ---------------------------------------------------------------------------
    built = sorted((c for c in clips if c.get("t_generated")), key=lambda c: c["t_generated"])
    prev = min(c["t_enqueue"] for c in clips) if clips else 0
    for c in built:
        c["build_s"] = c["t_generated"] - max(prev, c["t_enqueue"])
        prev = c["t_generated"]
        c["enqueue_to_built_s"] = c["t_generated"] - c["t_enqueue"]
        if c.get("t_started"):
            c["enqueue_to_first_frame_s"] = c["t_started"] - c["t_enqueue"]
    builds = [c["build_s"] for c in built]
    secs_built = sum(c.get("seconds") or args.seconds for c in built)
    span = (built[-1]["t_generated"] - min(c["t_enqueue"] for c in clips)) if built else None
    gaps = [b - a for a, b in zip(frame_times, frame_times[1:])]
    starts = sorted(c["t_started"] for c in clips if c.get("t_started"))
    ends = sorted(c["t_finished"] for c in clips if c.get("t_finished"))
    handoffs = [s - e for e, s in zip(ends, starts[1:])]  # gap between one clip ending and the next starting
    billed = (t_disc - t_ready) if t_ready and t_disc else None
    summary = {
        "run_id": run_id, "model": args.model, "clips": args.clips, "seconds_per_clip": args.seconds,
        "status_log": status_log,
        "time_to_ready_s": t_ready,
        "first_clip_built_s": clips[0].get("enqueue_to_built_s") if clips else None,
        "first_clip_first_frame_s": clips[0].get("enqueue_to_first_frame_s") if clips else None,
        "build_s_median": statistics.median(builds) if builds else None,
        "build_s_min": min(builds) if builds else None,
        "build_s_max": max(builds) if builds else None,
        "clips_built": len(built), "clips_failed": sum(1 for c in clips if c.get("t_failed")),
        "realtime_factor": secs_built / span if span else None,
        "frames_received": len(frame_times),
        "frame_gap_p95_ms": pct(gaps, .95) * 1000 if gaps else None,
        "clip_handoff_gap_s_median": statistics.median(handoffs) if handoffs else None,
        "billed_session_s": billed,
        "est_cost_usd": billed * RATE_PER_S if billed else None,
        "cost_per_output_s": (billed * RATE_PER_S / secs_built) if billed and secs_built else None,
        "session_state": {k: state.get(k) for k in ("clip_seconds_min", "clip_seconds_max", "generation_capacity",
                                                    "playout_capacity", "seconds_sent")},
    }
    os.makedirs(os.path.join(ROOT, "results", "raw"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "raw", f"{tag}_{run_id}.json"), "w") as f:
        json.dump({**summary, "per_clip": clips, "messages": messages[:300]}, f, indent=1, default=str)
    for k, v in summary.items():
        if k != "status_log":
            print(f"{k:28s} {round(v, 3) if isinstance(v, float) else v}")
    print("per clip: build_s, enqueue->built, enqueue->first frame, frames seen")
    for c in clips:
        print("  ", c["i"], *(round(c[k], 2) if isinstance(c.get(k), float) else c.get(k)
                              for k in ("build_s", "enqueue_to_built_s", "enqueue_to_first_frame_s", "frames_seen")))
    print(f"frames saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL,
                    help="any Reactor clip-queue model, e.g. reactor/h3-reference-to-video-turbo-realtime")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=5.167, help="FastH3 minimum; matches MachGen H3 5 s output (5.17 s)")
    ap.add_argument("--max-session", type=int, default=240, help="hard cap on session length, seconds")
    asyncio.run(main(ap.parse_args()))
