"""
ComfyUI API Bridge - txt2img and img2img support
OpenAI compatible image generation API
"""

import os
import json
import uuid
import time
import base64
import asyncio
import aiohttp
import io
from pathlib import Path
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# ==================== CONFIG ====================
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "10.126.3.166")
COMFYUI_PORT = os.getenv("COMFYUI_PORT", "8188")
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

API_KEY = os.getenv("API_KEY", "sk-comfyui-z-image-turbo")

WORKFLOW_TXT2IMG = os.getenv("WORKFLOW_TXT2IMG", "workflow_txt2img.json")
WORKFLOW_IMG2IMG = os.getenv("WORKFLOW_IMG2IMG", "workflow_img2img.json")

MODEL_NAME = "z-image-turbo"

# ==================== Request Models ====================
class ImageGenerationRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: str = "b64_json"
    quality: str = "standard"
    style: str = "natural"

class Img2ImgRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: str
    image: str  # base64 encoded image
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    response_format: str = "b64_json"

class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None

class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]

# ==================== Workflow Manager ====================
class WorkflowManager:
    def __init__(self):
        self.txt2img_template = None
        self.img2img_template = None
        self.load_workflows()

    def load_workflows(self):
        try:
            with open(WORKFLOW_TXT2IMG, 'r', encoding='utf-8') as f:
                self.txt2img_template = json.load(f)
            print(f"[OK] txt2img loaded: {WORKFLOW_TXT2IMG}")
        except FileNotFoundError:
            print(f"[WARN] txt2img not found: {WORKFLOW_TXT2IMG}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] txt2img JSON error: {e}")

        try:
            with open(WORKFLOW_IMG2IMG, 'r', encoding='utf-8') as f:
                self.img2img_template = json.load(f)
            print(f"[OK] img2img loaded: {WORKFLOW_IMG2IMG}")
        except FileNotFoundError:
            print(f"[WARN] img2img not found: {WORKFLOW_IMG2IMG}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] img2img JSON error: {e}")

    def find_prompt_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                meta = node.get("_meta", {})
                title = meta.get("title", "").lower()
                if any(kw in title for kw in ["正向", "positive", "prompt"]):
                    return node_id

        for node_id, node in workflow.items():
            if "KSampler" in node.get("class_type", ""):
                positive_input = node.get("inputs", {}).get("positive", [])
                if isinstance(positive_input, list) and len(positive_input) >= 1:
                    return str(positive_input[0])
        return None

    def find_sampler_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if "KSampler" in node.get("class_type", ""):
                return node_id
        return None

    def find_latent_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if node.get("class_type") == "EmptyLatentImage":
                return node_id
        return None

    def find_load_image_node(self, workflow: dict) -> Optional[str]:
        for node_id, node in workflow.items():
            if node.get("class_type") == "LoadImage":
                return node_id
        return None

    def prepare_txt2img(self, prompt: str, width: int, height: int, seed: int = None) -> dict:
        if self.txt2img_template is None:
            raise ValueError("txt2img workflow not loaded")

        workflow = json.loads(json.dumps(self.txt2img_template))

        prompt_node = self.find_prompt_node(workflow)
        if prompt_node and prompt_node in workflow:
            workflow[prompt_node]["inputs"]["text"] = prompt

        latent_node = self.find_latent_node(workflow)
        if latent_node and latent_node in workflow:
            workflow[latent_node]["inputs"]["width"] = width
            workflow[latent_node]["inputs"]["height"] = height

        if seed is None:
            seed = int(time.time() * 1000) % (2**32)
        sampler_node = self.find_sampler_node(workflow)
        if sampler_node and sampler_node in workflow:
            workflow[sampler_node]["inputs"]["seed"] = seed

        return workflow

    def prepare_img2img(self, prompt: str, image_name: str, strength: float, seed: int = None) -> dict:
        if self.img2img_template is None:
            raise ValueError("img2img workflow not loaded")

        workflow = json.loads(json.dumps(self.img2img_template))

        prompt_node = self.find_prompt_node(workflow)
        if prompt_node and prompt_node in workflow:
            workflow[prompt_node]["inputs"]["text"] = prompt

        load_image_node = self.find_load_image_node(workflow)
        if load_image_node and load_image_node in workflow:
            workflow[load_image_node]["inputs"]["image"] = image_name

        sampler_node = self.find_sampler_node(workflow)
        if sampler_node and sampler_node in workflow:
            workflow[sampler_node]["inputs"]["denoise"] = strength
            if seed is None:
                seed = int(time.time() * 1000) % (2**32)
            workflow[sampler_node]["inputs"]["seed"] = seed

        return workflow

# ==================== ComfyUI Client ====================
class ComfyUIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client_id = str(uuid.uuid4())

    async def upload_image(self, image_data: bytes, filename: str) -> str:
        form_data = aiohttp.FormData()
        form_data.add_field('image', image_data, filename=filename, content_type='image/png')
        form_data.add_field('overwrite', 'true')

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/upload/image", data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(status_code=500, detail=f"Upload failed: {error_text}")

                result = await response.json()
                return result.get("name", filename)

    async def queue_prompt(self, workflow: dict) -> str:
        payload = {
            "prompt": workflow,
            "client_id": self.client_id
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/prompt", json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(status_code=500, detail=f"ComfyUI error: {error_text}")

                result = await response.json()
                return result.get("prompt_id")

    async def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            while time.time() - start_time < timeout:
                async with session.get(f"{self.base_url}/history/{prompt_id}") as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            return history[prompt_id]

                await asyncio.sleep(0.5)

        raise HTTPException(status_code=504, detail="Generation timeout")

    async def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/view", params=params) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Failed to get image")
                return await response.read()

# ==================== FastAPI App ====================
workflow_manager: WorkflowManager = None
comfyui_client: ComfyUIClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global workflow_manager, comfyui_client
    workflow_manager = WorkflowManager()
    comfyui_client = ComfyUIClient(COMFYUI_URL)
    print(f"")
    print(f"========================================")
    print(f"  ComfyUI API Bridge v2.0")
    print(f"========================================")
    print(f"  ComfyUI:  {COMFYUI_URL}")
    print(f"  API Key:  {API_KEY[:15]}...")
    print(f"  txt2img:  {WORKFLOW_TXT2IMG}")
    print(f"  img2img:  {WORKFLOW_IMG2IMG}")
    print(f"========================================")
    yield
    print("API Bridge stopped")

app = FastAPI(
    title="ComfyUI API Bridge",
    description="OpenAI compatible API with txt2img and img2img",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    token = authorization.replace("Bearer ", "").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token

# ==================== API Endpoints ====================

@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    verify_api_key(authorization)
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": int(time.time()), "owned_by": "local"}]
    }

@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest, authorization: str = Header(None)):
    """Text to image (txt2img) - OpenAI compatible"""
    verify_api_key(authorization)

    if workflow_manager.txt2img_template is None:
        raise HTTPException(status_code=500, detail="txt2img workflow not configured")

    try:
        width, height = map(int, request.size.split("x"))
    except:
        width, height = 1024, 1024

    generated_images = []

    for i in range(request.n):
        seed = int(time.time() * 1000 + i) % (2**32)
        workflow = workflow_manager.prepare_txt2img(prompt=request.prompt, width=width, height=height, seed=seed)

        prompt_id = await comfyui_client.queue_prompt(workflow)
        print(f"[txt2img] Task: {prompt_id}")

        result = await comfyui_client.wait_for_completion(prompt_id)

        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    image_data = await comfyui_client.get_image(
                        filename=img_info["filename"],
                        subfolder=img_info.get("subfolder", ""),
                        folder_type=img_info.get("type", "output")
                    )
                    b64_data = base64.b64encode(image_data).decode("utf-8")
                    generated_images.append(ImageData(b64_json=b64_data, revised_prompt=request.prompt))
                    break
            if generated_images:
                break

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    return ImageGenerationResponse(created=int(time.time()), data=generated_images)

@app.post("/v1/images/edits", response_model=ImageGenerationResponse)
async def edit_images(request: Img2ImgRequest, authorization: str = Header(None)):
    """Image to image (img2img) - OpenAI compatible"""
    verify_api_key(authorization)

    if workflow_manager.img2img_template is None:
        raise HTTPException(status_code=500, detail="img2img workflow not configured")

    try:
        image_data = base64.b64decode(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    generated_images = []

    for i in range(request.n):
        filename = f"input_{int(time.time())}_{i}.png"
        uploaded_name = await comfyui_client.upload_image(image_data, filename)
        print(f"[img2img] Uploaded: {uploaded_name}")

        seed = int(time.time() * 1000 + i) % (2**32)
        workflow = workflow_manager.prepare_img2img(
            prompt=request.prompt, image_name=uploaded_name, strength=request.strength, seed=seed
        )

        prompt_id = await comfyui_client.queue_prompt(workflow)
        print(f"[img2img] Task: {prompt_id}")

        result = await comfyui_client.wait_for_completion(prompt_id)

        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    result_image = await comfyui_client.get_image(
                        filename=img_info["filename"],
                        subfolder=img_info.get("subfolder", ""),
                        folder_type=img_info.get("type", "output")
                    )
                    b64_data = base64.b64encode(result_image).decode("utf-8")
                    generated_images.append(ImageData(b64_json=b64_data, revised_prompt=request.prompt))
                    break
            if len(generated_images) > i:
                break

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    return ImageGenerationResponse(created=int(time.time()), data=generated_images)

@app.post("/v1/images/img2img", response_model=ImageGenerationResponse)
async def img2img_form(
        prompt: str = Form(...),
        image: UploadFile = File(...),
        strength: float = Form(0.6),
        size: str = Form("1024x1024"),
        n: int = Form(1),
        authorization: str = Header(None)
):
    """Image to image with form upload"""
    verify_api_key(authorization)

    if workflow_manager.img2img_template is None:
        raise HTTPException(status_code=500, detail="img2img workflow not configured")

    image_data = await image.read()
    generated_images = []

    for i in range(min(n, 4)):
        filename = f"input_{int(time.time())}_{i}.png"
        uploaded_name = await comfyui_client.upload_image(image_data, filename)
        print(f"[img2img-form] Uploaded: {uploaded_name}")

        seed = int(time.time() * 1000 + i) % (2**32)
        workflow = workflow_manager.prepare_img2img(
            prompt=prompt, image_name=uploaded_name, strength=strength, seed=seed
        )

        prompt_id = await comfyui_client.queue_prompt(workflow)
        print(f"[img2img-form] Task: {prompt_id}")

        result = await comfyui_client.wait_for_completion(prompt_id)

        outputs = result.get("outputs", {})
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    result_image = await comfyui_client.get_image(
                        filename=img_info["filename"],
                        subfolder=img_info.get("subfolder", ""),
                        folder_type=img_info.get("type", "output")
                    )
                    b64_data = base64.b64encode(result_image).decode("utf-8")
                    generated_images.append(ImageData(b64_json=b64_data, revised_prompt=prompt))
                    break
            if len(generated_images) > i:
                break

    if not generated_images:
        raise HTTPException(status_code=500, detail="Failed to generate image")

    return ImageGenerationResponse(created=int(time.time()), data=generated_images)

@app.get("/health")
async def health_check():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFYUI_URL}/system_stats", timeout=5) as response:
                comfyui_status = "ok" if response.status == 200 else "error"
    except:
        comfyui_status = "unreachable"

    return {
        "status": "ok",
        "comfyui": comfyui_status,
        "txt2img": workflow_manager.txt2img_template is not None,
        "img2img": workflow_manager.img2img_template is not None
    }

@app.post("/reload")
async def reload_workflows(authorization: str = Header(None)):
    verify_api_key(authorization)
    workflow_manager.load_workflows()
    return {
        "status": "ok",
        "txt2img": workflow_manager.txt2img_template is not None,
        "img2img": workflow_manager.img2img_template is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.126.3.102", port=8000)
