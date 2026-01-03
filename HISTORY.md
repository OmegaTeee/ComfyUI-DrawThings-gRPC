# Project History

## 2025-01-02

### Added
- `grpc_generate.py` CLI tool for image generation
  - Ollama prompt enhancement (`--enhance`)
  - LoRA support (`--lora flux-4step`)
  - ControlNet integration (`--control-image`, `--control-type`)
  - Upscaling options (`--upscale 2`)
  - Multiple models: flux-schnell, flux-dev, sd3, sdxl
- `grpc_upscale.py` CLI tool for batch upscaling
  - Supports ultrasharp, realesrgan-2x, realesrgan-4x, universal
  - Batch folder processing
- Documentation in Obsidian vault (`ComfyUI gRPC Integration.md`)

### Fixed
- Path references from `ComfyUI-source` to `ComfyUI` in:
  - `user/scripts/ollama_drawthings_generate.py`
  - `user/scripts/ollama_drawthings_grpc.py`
  - `user/scripts/OLLAMA_README.md`
- Removed stale `ComfyUI-source` directory

### Changed
- Updated `README.md` with CLI tools documentation
- Updated `custom_nodes/README.md` with usage examples

### Published
- Created GitHub repo: https://github.com/OmegaTeee/ComfyUI-DrawThings-gRPC
