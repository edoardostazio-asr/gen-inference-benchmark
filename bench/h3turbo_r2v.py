"""MiniMax H3 Turbo reference-to-video on MachGen and Reactor, same references, prompts and seeds.

    python -m bench.h3turbo_r2v --clips 6

MachGen:  model MiniMax-H3-Turbo, task R2V, reference = an earlier MachGen FLUX image (src_task_ids)
Reactor:  reactor/h3-reference-to-video-turbo-realtime, reference = the same JPEG, uploaded
Both run at the same time so they share network conditions and time of day. MachGen clips run one
at a time; Reactor builds its queue one clip at a time, so both sides do sequential work.
Clip length: 5 s requested on MachGen (5.17 s delivered), 5.167 s on Reactor (its minimum).
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from PIL import Image
from reactor_sdk import Reactor

from bench.providers import run_machgen
from bench.realtime_reactor import mint_jwt
from bench.suites import R2V_PROMPTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
REF_RUN = "20261012T183006Z"  # MachGen FLUX run whose images become the references
REACTOR_MODEL = "reactor/h3-reference-to-video-turbo-realtime"
REACTOR_RATE = 0.035  # USD per session second (api.reactor.inc/pricing)


def references(n):
    rows = [json.loads(line) for line in open(os.path.join(ROOT, "results", "raw", f"{REF_RUN}.jsonl"))]
    by_i = {r["i"]: r for r in rows if r["provider"] == "machgen" and r["status"] == "COMPLETED"}
    return [{"i": i, "task_id": by_i[i]["job_id"],
             "path": os.path.join(ROOT, "outputs", REF_RUN, f"machgen_flux2-dev_{i:03d}.jpg")} for i in range(n)]


async def run_machgen_side(refs, out_dir):
    recs = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=300)) as client:
        for ref in refs:
            suite = {"kind": "video", "machgen": {
                "model": "MiniMax-H3-Turbo", "task_type": "R2V", "src_task_ids": [ref["task_id"]],
                "video_config": {"duration_secs": 5, "height": 768, "aspect_ratio": "16:9"}}}
            rec = await run_machgen(client, "h3turbo-r2v", suite, ref["i"], R2V_PROMPTS[ref["i"]], 2000 + ref["i"], out_dir)
            print(f"  machgen #{ref['i']} {rec['status']} e2e={rec['t_download']} queue={rec['server_queue_s']} "
                  f"gpu={rec['server_gen_s']} {rec['error'] or ''}"[:160])
            recs.append(rec)
    return recs


async def run_reactor_side(refs, out_dir, max_session):
    reactor = Reactor(model_name=REACTOR_MODEL, jwt=mint_jwt(REACTOR_MODEL, max_session))
    t0 = time.perf_counter()
    now = lambda: time.perf_counter() - t0  # noqa: E731
    ready, done = asyncio.Event(), asyncio.Event()
    clips = [{"i": r["i"], "prompt": R2V_PROMPTS[r["i"]], "seed": 2000 + r["i"]} for r in refs]
    playing = {"i": None, "frames": 0}
    status_log, errors = [], []

    def on_status(s):
        status_log.append((round(now(), 3), str(s)))
        if str(s) == "ready":
            ready.set()

    def on_message(m):
        t = now()
        kind = m.get("type") if isinstance(m, dict) else None
        data = (m.get("data") or {}) if isinstance(m, dict) else {}
        if kind and ("error" in kind or kind == "clip_failed"):
            errors.append((round(t, 2), kind, str(data)[:300]))
        meta = (data.get("clip") or {}).get("metadata")
        if not meta or not str(meta).startswith("job:"):
            return
        c = clips[int(str(meta)[4:])]
        if kind == "clip_queued":
            c.update(t_queued=t, seconds=(data.get("clip") or {}).get("seconds"))
        elif kind == "clip_generated":
            c["t_generated"] = t
        elif kind == "clip_started":
            c["t_started"] = t
            playing.update(i=c["i"], frames=0)
        elif kind in ("clip_finished", "clip_stopped"):
            c["t_finished"] = t
        elif kind == "clip_failed":
            c["t_failed"] = t
        if all(cl.get("t_finished") or cl.get("t_failed") for cl in clips):
            done.set()

    reactor.on("status_changed", on_status)
    reactor.on("message", on_message)
    t_ready = None
    try:
        await reactor.connect()
        await asyncio.wait_for(ready.wait(), timeout=300)
        t_ready = now()

        @reactor.track("main_video").on_frame
        def _frame(rgb):
            i = playing["i"]
            if i is None:
                return
            playing["frames"] += 1
            clips[i]["frames_seen"] = playing["frames"]
            if playing["frames"] == 62:  # middle frame of a 124-frame clip
                Image.fromarray(rgb).save(os.path.join(out_dir, f"reactor_clip_{i}.jpg"), quality=92)

        uploads = []
        for r in refs:  # uploads happen before the clock starts for each clip, like MachGen's references
            uploads.append(await reactor.upload_file(r["path"]))
        await reactor.send_command("set_clip_seconds", {"seconds": 5.167})
        await reactor.send_command("set_autoplay", {"enabled": True})
        for c, ref in zip(clips, uploads):
            c["t_enqueue"] = now()
            await reactor.send_command("enqueue", {"prompt": c["prompt"], "seed": c["seed"], "seconds": 5.167,
                                                   "reference_image": ref, "metadata": f"job:{c['i']}"})
        try:
            await asyncio.wait_for(done.wait(), timeout=max_session - 30)
        except asyncio.TimeoutError:
            print("reactor: timed out waiting for clips")
    finally:
        await reactor.disconnect()  # releases the GPU and stops billing
        t_disc = now()
    built = sorted((c for c in clips if c.get("t_generated")), key=lambda c: c["t_generated"])
    prev = min(c["t_enqueue"] for c in clips if c.get("t_enqueue")) if built else 0
    for c in built:
        c["build_s"] = c["t_generated"] - max(prev, c["t_enqueue"])
        c["enqueue_to_built_s"] = c["t_generated"] - c["t_enqueue"]
        prev = c["t_generated"]
        print(f"  reactor #{c['i']} build={c['build_s']:.2f}s enqueue->built={c['enqueue_to_built_s']:.2f}s")
    billed = (t_disc - t_ready) if t_ready else None
    return {"status_log": status_log, "errors": errors, "clips": clips, "time_to_ready_s": t_ready,
            "billed_session_s": billed, "est_cost_usd": billed * REACTOR_RATE if billed else None}


async def main(args):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(ROOT, "outputs", f"h3turbo_r2v_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    refs = references(args.clips)
    mg, rx = await asyncio.gather(run_machgen_side(refs, out_dir), run_reactor_side(refs, out_dir, args.max_session))
    path = os.path.join(ROOT, "results", "raw", f"h3turbo_r2v_{run_id}.json")
    with open(path, "w") as f:
        json.dump({"run_id": run_id, "refs": refs, "machgen": mg, "reactor": rx}, f, indent=1, default=str)
    print(f"\nRaw: {path}\nOutputs: {out_dir}\nReactor errors: {rx['errors'][:3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--max-session", type=int, default=300, help="hard cap on the Reactor session, seconds")
    asyncio.run(main(ap.parse_args()))
