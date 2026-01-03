## ComfyUI-DrawThings-gRPC

**ComfyUI-DrawThings-gRPC** is a bridge between [ComfyUI](https://www.comfy.org/) and [Draw Things](https://drawthings.ai/) via gRPC. It allows ComfyUI to build and send image generation requests to Draw Things - giving you more control over inputs and settings than Draw Things alone offers, and bringing the Draw Things sampler into your ComfyUI workflows.

---

### Setup

**Requirements**

- [ComfyUI](https://www.comfy.org/)
- [Draw Things](https://drawthings.ai/) (with gRPC server enabled) **or** [gRPCServerCLI](https://github.com/drawthingsai/draw-things-community/tree/main?tab=readme-ov-file#self-host-grpcservercli-from-packaged-binaries)


**Via ComfyUI-Manager (Recommended)**
- Search for `ComfyUI-DrawThings-gRPC` in the ComfyUI-Manager and install.

**Manual Installation**
- Clone this repository into your `ComfyUI/custom_nodes` directory:
  ```sh
  git clone https://github.com/yourusername/ComfyUI-DrawThings-gRPC.git
  ```

**Restart ComfyUI**

---

### Configuring Draw Things gRPC Server

#### Draw Things App

Ensure the following settings are enabled:
- **API Server:** enabled
- **Protocol:** gRPC
- **Transport Layer Security:** Enabled
- **Enable Model Browser:** Enabled
- **Response Compression:** Disabled


#### gRPCServerCLI

Start the server with:
```sh
gRPCServerCLI-macOS [path to models] --no-response-compression --model-browser
```

---

### CLI Tools

Two command-line tools are included for quick generation without the ComfyUI GUI.

#### grpc_generate.py

Generate images with full feature support including LoRA, ControlNet, upscaling, and Ollama prompt enhancement.

```bash
# Basic generation
./grpc_generate.py "a red fox in snow" --save

# With Ollama prompt enhancement
./grpc_generate.py "a cat" --enhance --save

# With LoRA and upscaling
./grpc_generate.py "portrait" --lora flux-4step --upscale 2 --save

# With ControlNet
./grpc_generate.py "landscape" --control-image ref.png --control-type Depth
```

**Options:**
- `--model, -m` - Model: flux-schnell (default), flux-dev, sd3, sdxl
- `--enhance, -e` - Use Ollama to enhance prompt
- `--lora, -l` - LoRA: flux-4step, sdxl-offset
- `--upscale, -u` - Upscale factor: 1-4
- `--width, -W` / `--height, -H` - Image dimensions
- `--save` - Save to output folder

#### grpc_upscale.py

Batch upscale images using Draw Things upscalers.

```bash
# Single image
./grpc_upscale.py image.png -s 2

# Folder of images
./grpc_upscale.py folder/

# Specific upscaler
./grpc_upscale.py images/ -u realesrgan-4x -s 4
```

**Available upscalers:** ultrasharp, realesrgan-2x, realesrgan-4x, universal

---

### Discussion

Join the conversation and get support on [Discord](https://discord.com/channels/1038516303666876436/1357377020299837464).

---

### Acknowledgements

- [draw-things-community](https://github.com/drawthingsai/draw-things-community)
- [ComfyUI-DrawThingsWrapper](https://github.com/JosephThomasParker/ComfyUI-DrawThingsWrapper)
- [dt-grpc-ts](https://github.com/kcjerrell/dt-grpc-ts)
- [ComfyUI_tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes)

> **Note:**
> This project was created with the [cookiecutter-comfy-extension](https://github.com/Comfy-Org/cookiecutter-comfy-extension) template to simplify custom node development for ComfyUI.
