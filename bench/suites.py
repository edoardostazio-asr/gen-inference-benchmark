"""Matched model pairs. Each suite runs the same weights at the same output settings on every provider.

Settings left to each provider's default (and therefore part of what is being compared) are listed
in `provider_defaults` so the report can state them.
"""

PROMPTS = [
    "A red fox trotting through fresh snow in a birch forest at dawn, soft golden light",
    "A hot air balloon drifting over a lavender field in light rain, slow tilt up, cinematic",
    "A barista pouring latte art in a sunlit cafe, shallow depth of field, close-up",
    "A lighthouse on a rocky cliff during a storm, waves crashing, dramatic clouds",
    "A golden retriever puppy chasing bubbles in a backyard garden, slow motion",
    "A futuristic city street at night with neon signs reflected in puddles, light fog",
    "An astronaut walking across red desert dunes on Mars, wide shot, dust in the air",
    "A paper boat floating down a rain-filled gutter, macro shot, reflections",
    "A chef tossing vegetables in a flaming wok in a busy restaurant kitchen",
    "A hummingbird hovering at a bright pink flower, background softly blurred",
]

# Reference-to-video prompts, one per reference image (the FLUX outputs for PROMPTS[0..5]).
R2V_PROMPTS = [
    "The fox from the reference image trots through the snowy birch forest toward the camera, slow tracking shot, soft crunch of snow",
    "The hot air balloon from the reference image rises slowly over the lavender field, gentle rain, camera tilts up, wind and distant birds",
    "The barista pours milk into the cup from the reference image, latte art forming, close-up, cafe chatter and milk steamer hiss",
    "Waves crash against the cliff below the lighthouse from the reference image, storm clouds move fast, rain lashes, thunder rumbles",
    "The puppy from the reference image leaps to catch a floating bubble in the garden, slow motion, playful barks",
    "The camera glides down the neon street from the reference image at night, rain falls, puddles ripple, city hum and passing cars",
]

SUITES = {
    # --- Images -------------------------------------------------------------------------------
    "flux2-dev": {
        "kind": "image",
        "machgen": {
            "model": "FLUX.2-dev",
            "task_type": "T2I",
            "image_config": {"width": 1024, "height": 1024, "infer_steps": 28, "guidance_scale": [2.5]},
        },
        "fal": {
            "endpoint": "fal-ai/flux-2",
            "args": {"image_size": {"width": 1024, "height": 1024}, "num_inference_steps": 28,
                     "guidance_scale": 2.5, "output_format": "jpeg", "enable_safety_checker": False},
        },
        # USD per job, used for the budget guard. MachGen $0.004/MP, fal $0.012/MP.
        "est_cost": {"machgen": 0.0042, "fal": 0.0126},
        "provider_defaults": "MachGen optimization_level=auto; fal acceleration=regular",
    },
    # --- Video --------------------------------------------------------------------------------
    "minimax-h3-768p-5s": {
        "kind": "video",
        "machgen": {
            "model": "MiniMax-H3",
            "task_type": "T2V",
            "video_config": {"duration_secs": 5, "height": 768, "aspect_ratio": "16:9"},
        },
        "fal": {
            "endpoint": "minimax/h3/text-to-video",
            "args": {"duration": 5, "resolution": "768P", "aspect_ratio": "16:9", "enable_safety_checker": False},
        },
        "est_cost": {"machgen": 0.0875, "fal": 0.25},
        "provider_defaults": "both providers run their default prompt enhancer",
    },
    "minimax-h3-turbo-768p-5s": {
        "kind": "video",
        "machgen": {
            "model": "MiniMax-H3-Turbo",
            "task_type": "T2V",
            "video_config": {"duration_secs": 5, "height": 768, "aspect_ratio": "16:9"},
        },
        "fal": {
            "endpoint": "minimax/h3-max-turbo/text-to-video",
            "args": {"duration": 5, "resolution": "768P", "aspect_ratio": "16:9", "enable_safety_checker": False},
        },
        # MachGen promo $0.01/s (list $0.02/s); fal side unused in the MachGen-vs-Reactor run.
        "est_cost": {"machgen": 0.10, "fal": 0.25},
        "provider_defaults": "MachGen default turbo adapter",
    },
    "ltx23-pro-1080p-6s": {
        "kind": "video",
        "machgen": {
            "model": "LTX-2.3-Pro",
            "task_type": "T2V",
            "video_config": {"duration_secs": 6, "height": 1080, "aspect_ratio": "16:9", "fps": 25, "audio": True},
        },
        "fal": {
            "endpoint": "fal-ai/ltx-2.3/text-to-video",
            "args": {"duration": 6, "resolution": "1080p", "aspect_ratio": "16:9", "fps": 25,
                     "generate_audio": True, "enable_safety_checker": False},
        },
        "est_cost": {"machgen": 0.048, "fal": 0.48},
        "provider_defaults": "MachGen prompt enhancer on by default",
    },
    "wan22-a14b-720p-5s": {
        "kind": "video",
        "machgen": {
            "model": "Wan2.2-A14B",
            "task_type": "T2V",
            "video_config": {"duration_secs": 5, "height": 720, "aspect_ratio": "16:9", "fps": 16, "infer_steps": 27},
        },
        "fal": {
            "endpoint": "fal-ai/wan/v2.2-a14b/text-to-video",
            "args": {"num_frames": 81, "frames_per_second": 16, "resolution": "720p", "aspect_ratio": "16:9",
                     "num_inference_steps": 27, "interpolator_model": "none", "enable_safety_checker": False},
        },
        "est_cost": {"machgen": 0.09, "fal": 0.40},
        "provider_defaults": "fal acceleration=regular",
    },
}
