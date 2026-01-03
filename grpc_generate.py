#!/usr/bin/env python3
"""
Draw Things gRPC Generator CLI

Generate images using Draw Things gRPC API through ComfyUI.
Supports LoRA, ControlNet, Upscaler, and Ollama prompt enhancement.

Usage:
    grpc_generate.py "a dragon in a castle"
    grpc_generate.py "prompt" --enhance              # Use Ollama to enhance prompt
    grpc_generate.py "prompt" --lora flux-4step --upscale 2
    grpc_generate.py "prompt" --control-image ref.png --control-type Canny
"""

import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"
OLLAMA_URL = "http://127.0.0.1:11434"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"

# Ollama prompt enhancement settings
OLLAMA_MODEL = "llama3.2:latest"
OLLAMA_SYSTEM_PROMPT = """You are an expert prompt engineer for Stable Diffusion and Flux image generation.
Take the user's simple image description and transform it into a detailed, evocative prompt that will produce stunning results.
Add artistic style, lighting, atmosphere, camera angle, and technical details.
Keep the core subject but enhance with vivid details.
Reply ONLY with the enhanced prompt, no explanations or additional text."""


def enhance_prompt_with_ollama(prompt, quiet=False):
    """Enhance a prompt using Ollama LLM."""
    try:
        if not quiet:
            print(f"Enhancing prompt with Ollama...")

        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": OLLAMA_SYSTEM_PROMPT,
                "stream": False
            },
            timeout=60
        )

        if response.ok:
            result = response.json()
            enhanced = result.get("response", "").strip()
            if enhanced:
                if not quiet:
                    print(f"Enhanced: {enhanced[:70]}...")
                return enhanced

        if not quiet:
            print("Warning: Ollama enhancement failed, using original prompt")
        return prompt

    except requests.exceptions.ConnectionError:
        if not quiet:
            print("Warning: Ollama not running, using original prompt")
        return prompt
    except Exception as e:
        if not quiet:
            print(f"Warning: Ollama error ({e}), using original prompt")
        return prompt

# Available models (update based on your Draw Things installation)
MODELS = {
    "flux-schnell": {"file": "flux_1_schnell_q5p.ckpt", "version": "flux1", "steps": 8},
    "flux-dev": {"file": "flux_1_dev_q8p.ckpt", "version": "flux1", "steps": 20},
    "sd3": {"file": "sd3_medium_3.5_q8p.ckpt", "version": "sd3", "steps": 20},
    "sdxl": {"file": "dreamshaper_xl_v2.1_turbo_f16.ckpt", "version": "sdxl_base_v0.9", "steps": 8},
}

LORAS = {
    "flux-4step": {"file": "flux.1__dev__to__schnell__4_step_lora_f16.ckpt", "version": "flux1"},
    "sdxl-offset": {"file": "sdxl_offset_v1.0_lora_f16.ckpt", "version": "sdxl_base_v0.9"},
}

CONTROLNETS = {
    "flux-union": {"file": "controlnet_union_pro_flux_1_dev_1.0_q5p.ckpt", "version": "flux1"},
    "canny-v2": {"file": "controlnet_canny_2.x_f16.ckpt", "version": "v2"},
    "tile-v1": {"file": "controlnet_tile_1.x_v1.1_f16.ckpt", "version": "v1"},
}

UPSCALERS = {
    "ultrasharp": {"file": "4x_ultrasharp_f16.ckpt", "name": "4x UltraSharp"},
    "realesrgan-2x": {"file": "realesrgan_x2plus_f16.ckpt", "name": "Real-ESRGAN X2+"},
    "realesrgan-4x": {"file": "realesrgan_x4plus_f16.ckpt", "name": "Real-ESRGAN X4+"},
}

CONTROL_TYPES = [
    "Canny", "Depth", "Pose", "Scribble", "Color", "Lineart",
    "Softedge", "Tile", "Blur", "Gray", "Custom"
]


def build_workflow(args):
    """Build the ComfyUI workflow based on arguments."""
    workflow = {"prompt": {}}
    node_id = 1

    # Get model info
    model_info = MODELS.get(args.model, MODELS["flux-schnell"])
    steps = args.steps if args.steps else model_info["steps"]

    # Adjust steps if using 4-step LoRA
    if args.lora == "flux-4step":
        steps = 4

    # Load image node (if using ControlNet)
    control_link = None
    if args.control_image:
        workflow["prompt"][str(node_id)] = {
            "class_type": "LoadImage",
            "inputs": {"image": args.control_image}
        }
        image_node_id = node_id
        node_id += 1

        # ControlNet node
        cnet_info = CONTROLNETS.get(args.controlnet, CONTROLNETS["flux-union"])
        workflow["prompt"][str(node_id)] = {
            "class_type": "DrawThingsControlNet",
            "inputs": {
                "control_name": {"value": {"file": cnet_info["file"], "version": cnet_info["version"]}},
                "control_input_type": args.control_type,
                "control_mode": "Balanced",
                "control_weight": args.control_weight,
                "control_start": 0.0,
                "control_end": 0.8,
                "global_average_pooling": False,
                "down_sampling_rate": 1.0,
                "target_blocks": "All",
                "invert_image": False,
                "image": [str(image_node_id), 0]
            }
        }
        control_link = [str(node_id), 0]
        node_id += 1

    # LoRA node
    lora_link = None
    if args.lora:
        lora_info = LORAS.get(args.lora)
        if lora_info:
            workflow["prompt"][str(node_id)] = {
                "class_type": "DrawThingsLoRA",
                "inputs": {
                    "buttons": None,
                    "lora": {"value": {"file": lora_info["file"], "version": lora_info["version"]}},
                    "weight": args.lora_weight,
                    "mode": "All",
                    **{f"lora_{i}": None for i in range(2, 9)},
                    **{f"weight_{i}": 1.0 for i in range(2, 9)},
                    **{f"mode_{i}": "All" for i in range(2, 9)}
                }
            }
            lora_link = [str(node_id), 0]
            node_id += 1

    # Upscaler node
    upscaler_link = None
    if args.upscale:
        upscaler_info = UPSCALERS.get(args.upscaler, UPSCALERS["ultrasharp"])
        workflow["prompt"][str(node_id)] = {
            "class_type": "DrawThingsUpscaler",
            "inputs": {
                "upscaler_model": {"value": upscaler_info},
                "upscaler_scale_factor": args.upscale
            }
        }
        upscaler_link = [str(node_id), 0]
        node_id += 1

    # Main sampler node
    sampler_inputs = {
        "settings": "Basic",
        "server": args.server,
        "port": args.port,
        "use_tls": args.tls,
        "model": {"value": {"file": model_info["file"], "version": model_info["version"]}},
        "strength": 1.0,
        "seed": args.seed,
        "seed_mode": "ScaleAlike",
        "width": args.width,
        "height": args.height,
        "steps": steps,
        "num_frames": 1,
        "cfg": args.cfg,
        "cfg_zero_star": False,
        "cfg_zero_star_init_steps": 0,
        "speed_up": True,
        "guidance_embed": args.cfg,
        "sampler_name": args.sampler,
        "stochastic_sampling_gamma": 0.3,
        "res_dpt_shift": True,
        "shift": 1.0,
        "batch_size": 1,
        "fps": 5,
        "motion_scale": 127,
        "guiding_frame_noise": 0.02,
        "start_frame_guidance": 1.0,
        "causal_inference": 0,
        "causal_inference_pad": 0,
        "clip_skip": 1,
        "sharpness": 0.6,
        "mask_blur": 1.5,
        "mask_blur_outset": 0,
        "preserve_original": True,
        "high_res_fix": False,
        "high_res_fix_start_width": 448,
        "high_res_fix_start_height": 448,
        "high_res_fix_strength": 0.7,
        "tiled_decoding": False,
        "decoding_tile_width": 640,
        "decoding_tile_height": 640,
        "decoding_tile_overlap": 128,
        "tiled_diffusion": False,
        "diffusion_tile_width": 512,
        "diffusion_tile_height": 512,
        "diffusion_tile_overlap": 64,
        "tea_cache": False,
        "tea_cache_start": 5,
        "tea_cache_end": 2,
        "tea_cache_threshold": 0.2,
        "tea_cache_max_skip_steps": 3,
        "separate_clip_l": False,
        "clip_l_text": "",
        "separate_open_clip_g": False,
        "open_clip_g_text": "",
        "positive": args.prompt,
        "negative": args.negative,
    }

    if lora_link:
        sampler_inputs["lora"] = lora_link
    if control_link:
        sampler_inputs["control_net"] = control_link
    if upscaler_link:
        sampler_inputs["upscaler"] = upscaler_link

    workflow["prompt"][str(node_id)] = {
        "class_type": "DrawThingsSampler",
        "inputs": sampler_inputs
    }
    sampler_node_id = node_id
    node_id += 1

    # Preview/Save node
    workflow["prompt"][str(node_id)] = {
        "class_type": "SaveImage" if args.save else "PreviewImage",
        "inputs": {
            "images": [str(sampler_node_id), 0],
            **({"filename_prefix": args.output} if args.save else {})
        }
    }

    return workflow


def wait_for_completion(prompt_id, timeout=300):
    """Wait for the prompt to complete."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
            if response.ok:
                history = response.json()
                if prompt_id in history:
                    return history[prompt_id]
        except:
            pass
        time.sleep(1)
    return None


def get_output_images(history):
    """Extract output image paths from history."""
    images = []
    if history and "outputs" in history:
        for node_id, output in history["outputs"].items():
            if "images" in output:
                for img in output["images"]:
                    if img.get("type") == "output":
                        images.append(OUTPUT_DIR / img["filename"])
                    elif img.get("type") == "temp":
                        images.append(Path(__file__).parent.parent.parent / "temp" / img["filename"])
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Draw Things gRPC API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "a dragon in a castle"
  %(prog)s "a cat" --enhance                    # Ollama prompt enhancement
  %(prog)s "portrait" --model flux-dev --lora flux-4step
  %(prog)s "landscape" --control-image ref.png --control-type Depth
  %(prog)s "character" --upscale 2 --upscaler ultrasharp
  %(prog)s "scene" -e --lora flux-4step -u 2    # Full combo with enhancement

Available models: """ + ", ".join(MODELS.keys()) + """
Available LoRAs: """ + ", ".join(LORAS.keys()) + """
Available ControlNets: """ + ", ".join(CONTROLNETS.keys()) + """
Available Upscalers: """ + ", ".join(UPSCALERS.keys()) + """
Control types: """ + ", ".join(CONTROL_TYPES)
    )

    parser.add_argument("prompt", help="Image generation prompt")

    # Model options
    parser.add_argument("--model", "-m", default="flux-schnell",
                        choices=list(MODELS.keys()), help="Model to use")
    parser.add_argument("--steps", "-s", type=int, help="Number of steps (auto if not set)")
    parser.add_argument("--cfg", "-g", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--sampler", default="Euler A Trailing", help="Sampler name")

    # Size options
    parser.add_argument("--width", "-W", type=int, default=512, help="Image width")
    parser.add_argument("--height", "-H", type=int, default=512, help="Image height")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")

    # LoRA options
    parser.add_argument("--lora", "-l", choices=list(LORAS.keys()), help="LoRA to apply")
    parser.add_argument("--lora-weight", type=float, default=1.0, help="LoRA weight")

    # ControlNet options
    parser.add_argument("--control-image", "-c", help="Control image path")
    parser.add_argument("--control-type", default="Depth", choices=CONTROL_TYPES, help="Control type")
    parser.add_argument("--control-weight", type=float, default=0.7, help="ControlNet weight")
    parser.add_argument("--controlnet", default="flux-union", choices=list(CONTROLNETS.keys()),
                        help="ControlNet model")

    # Upscaler options
    parser.add_argument("--upscale", "-u", type=int, choices=[1, 2, 3, 4], help="Upscale factor")
    parser.add_argument("--upscaler", default="ultrasharp", choices=list(UPSCALERS.keys()),
                        help="Upscaler model")

    # Output options
    parser.add_argument("--negative", "-n", default="ugly, blurry, low quality, distorted",
                        help="Negative prompt")
    parser.add_argument("--output", "-o", default="grpc_output", help="Output filename prefix")
    parser.add_argument("--save", action="store_true", help="Save to output folder (vs preview)")

    # Server options
    parser.add_argument("--server", default="localhost", help="gRPC server address")
    parser.add_argument("--port", default="7860", help="gRPC server port")
    parser.add_argument("--no-tls", dest="tls", action="store_false", help="Disable TLS")

    # Ollama enhancement
    parser.add_argument("--enhance", "-e", action="store_true",
                        help="Enhance prompt using Ollama LLM")
    parser.add_argument("--ollama-model", default="llama3.2:latest",
                        help="Ollama model for enhancement")

    # Other options
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

    args = parser.parse_args()

    # Check ComfyUI is running
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
    except:
        print("Error: ComfyUI is not running on", COMFYUI_URL)
        print("Start it with: comfyui")
        sys.exit(1)

    # Enhance prompt with Ollama if requested
    original_prompt = args.prompt
    if args.enhance:
        global OLLAMA_MODEL
        OLLAMA_MODEL = args.ollama_model
        args.prompt = enhance_prompt_with_ollama(args.prompt, args.quiet)

    # Build and submit workflow
    workflow = build_workflow(args)

    if not args.quiet:
        if args.enhance and args.prompt != original_prompt:
            print(f"Original: {original_prompt[:50]}...")
        print(f"Generating: {args.prompt[:50]}...")
        features = []
        if args.enhance:
            features.append("Ollama")
        if args.lora:
            features.append(f"LoRA:{args.lora}")
        if args.control_image:
            features.append(f"ControlNet:{args.control_type}")
        if args.upscale:
            features.append(f"Upscale:{args.upscale}x")
        if features:
            print(f"Features: {', '.join(features)}")
        print(f"Size: {args.width}x{args.height}", end="")
        if args.upscale:
            print(f" → {args.width * args.upscale}x{args.height * args.upscale}", end="")
        print()

    try:
        response = requests.post(f"{COMFYUI_URL}/prompt", json=workflow)
        result = response.json()

        if "error" in result:
            print("Error:", result["error"]["message"])
            if "node_errors" in result:
                for node_id, errors in result["node_errors"].items():
                    for err in errors.get("errors", []):
                        print(f"  Node {node_id}: {err['message']}")
            sys.exit(1)

        prompt_id = result["prompt_id"]
        if not args.quiet:
            print(f"Queued: {prompt_id}")
            print("Generating...", end="", flush=True)

        # Wait for completion
        history = wait_for_completion(prompt_id, args.timeout)

        if history:
            images = get_output_images(history)
            if not args.quiet:
                print(" Done!")
            for img in images:
                print(f"Output: {img}")
        else:
            print("\nTimeout waiting for generation")
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to ComfyUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
