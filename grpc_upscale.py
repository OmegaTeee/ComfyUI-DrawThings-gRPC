#!/usr/bin/env python3
"""
Draw Things gRPC Batch Upscaler

Upscale images using Draw Things gRPC API through ComfyUI.

Usage:
    grpc_upscale.py image.png                    # Upscale single image 2x
    grpc_upscale.py folder/                      # Upscale all images in folder
    grpc_upscale.py *.jpg --scale 4              # Upscale with 4x
    grpc_upscale.py folder/ -u ultrasharp -s 2   # Specific upscaler at 2x
"""

import argparse
import json
import os
import sys
import time
import glob
import requests
from pathlib import Path

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
INPUT_DIR = Path(__file__).parent.parent.parent / "input"

UPSCALERS = {
    "ultrasharp": {"file": "4x_ultrasharp_f16.ckpt", "name": "4x UltraSharp"},
    "realesrgan-2x": {"file": "realesrgan_x2plus_f16.ckpt", "name": "Real-ESRGAN X2+"},
    "realesrgan-4x": {"file": "realesrgan_x4plus_f16.ckpt", "name": "Real-ESRGAN X4+"},
    "universal": {"file": "esrgan_4x_universal_upscaler_v2_sharp_f16.ckpt", "name": "Universal 4x"},
}


def get_image_dimensions(image_path):
    """Get image width and height."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def copy_to_input(image_path):
    """Copy image to ComfyUI input folder and return the filename and dimensions."""
    src = Path(image_path).resolve()
    dst = INPUT_DIR / src.name

    # Get dimensions before copying
    width, height = get_image_dimensions(src)

    # Avoid copying if already in input folder
    if src.parent == INPUT_DIR:
        return src.name, width, height

    # Copy file
    import shutil
    shutil.copy2(src, dst)
    return dst.name, width, height


def build_upscale_workflow(image_name, args, width, height):
    """Build the ComfyUI workflow for upscaling."""
    upscaler_info = UPSCALERS.get(args.upscaler, UPSCALERS["ultrasharp"])

    workflow = {
        "prompt": {
            # Load the image
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name}
            },
            # ControlNet in Color mode to pass through the image
            "2": {
                "class_type": "DrawThingsControlNet",
                "inputs": {
                    "control_name": {"value": {"file": "controlnet_union_pro_flux_1_dev_1.0_q5p.ckpt", "version": "flux1"}},
                    "control_input_type": "Color",
                    "control_mode": "Control",
                    "control_weight": 1.0,
                    "control_start": 0.0,
                    "control_end": 1.0,
                    "global_average_pooling": False,
                    "down_sampling_rate": 1.0,
                    "target_blocks": "All",
                    "invert_image": False,
                    "image": ["1", 0]
                }
            },
            # Upscaler node
            "3": {
                "class_type": "DrawThingsUpscaler",
                "inputs": {
                    "upscaler_model": {"value": upscaler_info},
                    "upscaler_scale_factor": args.scale
                }
            },
            # Sampler with minimal generation
            "4": {
                "class_type": "DrawThingsSampler",
                "inputs": {
                    "settings": "Basic",
                    "server": args.server,
                    "port": args.port,
                    "use_tls": args.tls,
                    "model": {"value": {"file": "flux_1_schnell_q5p.ckpt", "version": "flux1"}},
                    "strength": 0.1,  # Very low to preserve original
                    "seed": 42,
                    "seed_mode": "ScaleAlike",
                    "width": width,
                    "height": height,
                    "steps": 1,  # Single step
                    "num_frames": 1,
                    "cfg": 1.0,
                    "cfg_zero_star": False,
                    "cfg_zero_star_init_steps": 0,
                    "speed_up": True,
                    "guidance_embed": 1.0,
                    "sampler_name": "Euler A Trailing",
                    "stochastic_sampling_gamma": 0.0,
                    "res_dpt_shift": True,
                    "shift": 1.0,
                    "batch_size": 1,
                    "fps": 5,
                    "motion_scale": 127,
                    "guiding_frame_noise": 0.0,
                    "start_frame_guidance": 1.0,
                    "causal_inference": 0,
                    "causal_inference_pad": 0,
                    "clip_skip": 1,
                    "sharpness": 0.0,
                    "mask_blur": 0.0,
                    "mask_blur_outset": 0,
                    "preserve_original": True,
                    "high_res_fix": False,
                    "high_res_fix_start_width": 448,
                    "high_res_fix_start_height": 448,
                    "high_res_fix_strength": 0.0,
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
                    "positive": "high quality, detailed",
                    "negative": "",
                    "control_net": ["2", 0],
                    "upscaler": ["3", 0],
                    "image": ["1", 0]
                }
            },
            # Save image
            "5": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["4", 0],
                    "filename_prefix": args.prefix
                }
            }
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
        time.sleep(0.5)
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
    return images


def get_image_files(paths):
    """Get list of image files from paths (files, folders, or globs)."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
    images = []

    for path in paths:
        p = Path(path)
        if p.is_dir():
            # Get all images in directory
            for ext in image_extensions:
                images.extend(p.glob(f"*{ext}"))
                images.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in image_extensions:
            images.append(p)
        elif '*' in path or '?' in path:
            # Handle glob patterns
            for match in glob.glob(path):
                mp = Path(match)
                if mp.is_file() and mp.suffix.lower() in image_extensions:
                    images.append(mp)

    return sorted(set(images))


def main():
    parser = argparse.ArgumentParser(
        description="Batch upscale images using Draw Things gRPC API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                      # Upscale single image 2x
  %(prog)s folder/                        # Upscale all images in folder
  %(prog)s *.jpg -s 4                     # Upscale JPGs 4x
  %(prog)s folder/ -u realesrgan-4x       # Use Real-ESRGAN
  %(prog)s img.png -s 2 -p upscaled       # Custom output prefix

Available upscalers: """ + ", ".join(UPSCALERS.keys())
    )

    parser.add_argument("images", nargs="+", help="Image files, folders, or glob patterns")
    parser.add_argument("--scale", "-s", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Upscale factor (default: 2)")
    parser.add_argument("--upscaler", "-u", default="ultrasharp", choices=list(UPSCALERS.keys()),
                        help="Upscaler model (default: ultrasharp)")
    parser.add_argument("--prefix", "-p", default="upscaled", help="Output filename prefix")

    # Server options
    parser.add_argument("--server", default="localhost", help="gRPC server address")
    parser.add_argument("--port", default="7860", help="gRPC server port")
    parser.add_argument("--no-tls", dest="tls", action="store_false", help="Disable TLS")

    parser.add_argument("--timeout", type=int, default=300, help="Timeout per image in seconds")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Check ComfyUI is running
    try:
        requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
    except:
        print("Error: ComfyUI is not running on", COMFYUI_URL)
        print("Start it with: comfyui")
        sys.exit(1)

    # Get list of images
    images = get_image_files(args.images)

    if not images:
        print("No images found to upscale")
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(images)} image(s) to upscale {args.scale}x with {args.upscaler}")
        print()

    # Ensure input directory exists
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each image
    success = 0
    failed = 0

    for i, image_path in enumerate(images, 1):
        if not args.quiet:
            print(f"[{i}/{len(images)}] {image_path.name}...", end=" ", flush=True)

        try:
            # Copy image to input folder and get dimensions
            image_name, img_width, img_height = copy_to_input(image_path)

            # Build and submit workflow
            workflow = build_upscale_workflow(image_name, args, img_width, img_height)
            response = requests.post(f"{COMFYUI_URL}/prompt", json=workflow)
            result = response.json()

            if "error" in result:
                if not args.quiet:
                    print(f"Error: {result['error']['message']}")
                failed += 1
                continue

            prompt_id = result["prompt_id"]

            # Wait for completion
            history = wait_for_completion(prompt_id, args.timeout)

            if history:
                output_images = get_output_images(history)
                if output_images:
                    if not args.quiet:
                        print(f"-> {output_images[0].name}")
                    success += 1
                else:
                    if not args.quiet:
                        print("No output")
                    failed += 1
            else:
                if not args.quiet:
                    print("Timeout")
                failed += 1

        except Exception as e:
            if not args.quiet:
                print(f"Error: {e}")
            failed += 1

    # Summary
    if not args.quiet:
        print()
        print(f"Completed: {success}/{len(images)} images upscaled")
        if failed > 0:
            print(f"Failed: {failed}")
        print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
