"""
Hailo-to-Ollama API Gateway

This service exposes Ollama-compatible REST endpoints that translate
to HailoRT's LLM inference through the Python bindings.

Ollama API Reference: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import asyncio
import gc
import json
import os
import time
import uuid
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Hailo imports - assumes hailo_platform is installed
try:
    from hailo_platform.genai import VDevice, LLM, HailoSchedulingAlgorithm
    from hailo_platform import HailoSchedulingAlgorithm
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("WARNING: hailo_platform not available, running in mock mode")


# ============================================================================
# Pydantic Models (Ollama API Compatible)
# ============================================================================

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"


class ChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = True
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | List[str]
    options: Optional[Dict[str, Any]] = None


# ============================================================================
# Global State
# ============================================================================

class HailoState:
    def __init__(self):
        self.vdevice: Optional[Any] = None
        self.llm: Optional[Any] = None
        self.model_name: str = ""
        self.hef_path: str = ""
        self.lock = asyncio.Lock()

    def _release_device(self):
        """Explicitly release Hailo device resources."""
        old_llm = self.llm
        old_vdevice = self.vdevice
        self.llm = None
        self.vdevice = None
        self.model_name = ""
        self.hef_path = ""
        # Delete references and force GC to release native handles
        # before creating new ones, avoiding OUT_OF_PHYSICAL_DEVICES
        del old_llm
        del old_vdevice
        gc.collect()

    async def initialize(self, hef_path: str, model_name: str = "hailo-llm"):
        """Initialize Hailo device and LLM."""
        async with self.lock:
            if self.llm is not None and self.hef_path == hef_path:
                return  # Already initialized with same model

            self._release_device()

            if not os.path.isfile(hef_path):
                print(f"ERROR: HEF path is not a valid file: {hef_path}")
                return

            if HAILO_AVAILABLE:
                try:
                    params = VDevice.create_params()
                    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
                    params.group_id = "SHARED"
                    self.vdevice = VDevice(params)
                    self.llm = LLM(self.vdevice, hef_path)
                except Exception as e:
                    print(f"ERROR: Failed to initialize Hailo device: {e}")
                    self._release_device()
                    return

            self.hef_path = hef_path
            self.model_name = model_name

    def get_options(self, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract generation options from Ollama format."""
        if options is None:
            return {}

        # Map Ollama options to Hailo parameters
        hailo_options = {}
        if "temperature" in options:
            hailo_options["temperature"] = options["temperature"]
        if "top_p" in options:
            hailo_options["top_p"] = options["top_p"]
        if "top_k" in options:
            hailo_options["top_k"] = options["top_k"]
        if "num_predict" in options:
            hailo_options["max_generated_tokens"] = options["num_predict"]
        if "seed" in options:
            hailo_options["seed"] = options["seed"]
        if "repeat_penalty" in options:
            hailo_options["frequency_penalty"] = options["repeat_penalty"]

        return hailo_options


state = HailoState()


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize with default model if HEF_PATH env var is set
    hef_path = os.environ.get("HAILO_HEF_PATH", "")
    if hef_path:
        await state.initialize(hef_path)
    yield
    # Shutdown: explicitly release the Hailo device
    state._release_device()


app = FastAPI(
    title="Hailo Ollama Gateway",
    description="Ollama-compatible API gateway for Hailo AI accelerators",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Ollama-Compatible Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint (Ollama compatible)."""
    return "Hailo Ollama Gateway is running"


@app.head("/")
async def head_root():
    """HEAD health check (Ollama compatible)."""
    return JSONResponse(content={})


@app.get("/api/tags")
async def list_models():
    """List available models (Ollama /api/tags endpoint)."""
    models = []
    if state.model_name:
        models.append({
            "name": state.model_name,
            "model": state.model_name,
            "modified_at": "2025-01-01T00:00:00Z",
            "size": 0,
            "digest": "hailo",
            "details": {
                "parent_model": "",
                "format": "hef",
                "family": "hailo",
                "families": ["hailo"],
                "parameter_size": "unknown",
                "quantization_level": "hailo"
            }
        })

    return {"models": models}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    """
    Generate a response for a given prompt (Ollama /api/generate endpoint).

    Supports streaming (default) and non-streaming modes.
    """
    if state.llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set HAILO_HEF_PATH or call /api/pull first.")

    options = state.get_options(request.options)

    # Build prompt with system message if provided
    full_prompt = request.prompt
    if request.system:
        full_prompt = f"{request.system}\n\n{full_prompt}"

    if request.stream:
        return StreamingResponse(
            generate_stream(full_prompt, options, request.model),
            media_type="application/x-ndjson"
        )
    else:
        return await generate_full(full_prompt, options, request.model)


async def generate_stream(prompt: str, options: Dict[str, Any], model: str):
    """Stream tokens as NDJSON (Ollama format)."""
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    if HAILO_AVAILABLE and state.llm:
        full_response = ""
        try:
            with state.llm.generate(prompt, **options) as generator:
                for token in generator:
                    full_response += token
                    response = {
                        "model": model,
                        "created_at": created_at,
                        "response": token,
                        "done": False
                    }
                    yield json.dumps(response) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
            return

        # Final response with stats
        final = {
            "model": model,
            "created_at": created_at,
            "response": "",
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": len(prompt.split()),
            "prompt_eval_duration": 0,
            "eval_count": len(full_response.split()),
            "eval_duration": 0
        }
        yield json.dumps(final) + "\n"
    else:
        # Mock mode for testing
        mock_response = f"[Mock] Echo: {prompt[:100]}"
        for word in mock_response.split():
            response = {
                "model": model,
                "created_at": created_at,
                "response": word + " ",
                "done": False
            }
            yield json.dumps(response) + "\n"
            await asyncio.sleep(0.05)

        yield json.dumps({"model": model, "created_at": created_at, "response": "", "done": True}) + "\n"


async def generate_full(prompt: str, options: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Generate complete response (non-streaming)."""
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    if HAILO_AVAILABLE and state.llm:
        response_text = state.llm.generate_all(prompt, **options)
    else:
        response_text = f"[Mock] Echo: {prompt[:100]}"

    return {
        "model": model,
        "created_at": created_at,
        "response": response_text,
        "done": True,
        "done_reason": "stop",
        "context": [],
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": len(prompt.split()),
        "prompt_eval_duration": 0,
        "eval_count": len(response_text.split()),
        "eval_duration": 0
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Generate a chat response (Ollama /api/chat endpoint).

    Converts chat messages to a prompt using the model's chat template.
    """
    if state.llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Set HAILO_HEF_PATH or call /api/pull first.")

    options = state.get_options(request.options)

    # Build prompt from messages
    # Note: Hailo LLM may have its own chat template handling
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {msg.content}")

    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    if request.stream:
        return StreamingResponse(
            chat_stream(prompt, options, request.model),
            media_type="application/x-ndjson"
        )
    else:
        return await chat_full(prompt, options, request.model)


async def chat_stream(prompt: str, options: Dict[str, Any], model: str):
    """Stream chat response as NDJSON."""
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    if HAILO_AVAILABLE and state.llm:
        full_response = ""
        try:
            with state.llm.generate(prompt, **options) as generator:
                for token in generator:
                    full_response += token
                    response = {
                        "model": model,
                        "created_at": created_at,
                        "message": {
                            "role": "assistant",
                            "content": token
                        },
                        "done": False
                    }
                    yield json.dumps(response) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
            return

        final = {
            "model": model,
            "created_at": created_at,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": len(full_response.split()),
            "eval_duration": 0
        }
        yield json.dumps(final) + "\n"
    else:
        # Mock mode
        mock_response = "[Mock] Chat response"
        for word in mock_response.split():
            response = {
                "model": model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": word + " "},
                "done": False
            }
            yield json.dumps(response) + "\n"
            await asyncio.sleep(0.05)

        yield json.dumps({"model": model, "created_at": created_at, "message": {"role": "assistant", "content": ""}, "done": True}) + "\n"


async def chat_full(prompt: str, options: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Generate complete chat response (non-streaming)."""
    created_at = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    if HAILO_AVAILABLE and state.llm:
        response_text = state.llm.generate_all(prompt, **options)
    else:
        response_text = "[Mock] Chat response"

    return {
        "model": model,
        "created_at": created_at,
        "message": {
            "role": "assistant",
            "content": response_text
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": len(response_text.split()),
        "eval_duration": 0
    }


@app.post("/api/embeddings")
async def embeddings(request: EmbeddingsRequest):
    """
    Generate embeddings (Ollama /api/embeddings endpoint).

    Note: This requires the Hailo model to support embeddings.
    """
    # Hailo LLM may not directly support embeddings
    # This is a placeholder that returns an error
    raise HTTPException(
        status_code=501,
        detail="Embeddings not supported by Hailo LLM. Use a dedicated embedding model."
    )


@app.post("/api/pull")
async def pull_model(request: Request):
    """
    Pull/load a model (Ollama /api/pull endpoint).

    For Hailo, this expects a local HEF file path as the model name.
    """
    body = await request.json()
    model_name = body.get("name", "")

    # Treat model name as HEF path for Hailo
    if model_name.endswith(".hef"):
        try:
            await state.initialize(model_name, model_name)

            async def stream_status():
                yield json.dumps({"status": f"pulling {model_name}"}) + "\n"
                yield json.dumps({"status": "success"}) + "\n"

            return StreamingResponse(stream_status(), media_type="application/x-ndjson")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    else:
        raise HTTPException(
            status_code=400,
            detail="For Hailo, provide the full path to the .hef file as the model name"
        )


@app.get("/api/ps")
async def list_running():
    """List running models (Ollama /api/ps endpoint)."""
    models = []
    if state.llm is not None:
        models.append({
            "name": state.model_name,
            "model": state.model_name,
            "size": 0,
            "digest": "hailo",
            "expires_at": "2099-12-31T23:59:59Z",
            "size_vram": 0
        })

    return {"models": models}


@app.delete("/api/delete")
async def delete_model(request: Request):
    """Unload model (Ollama /api/delete endpoint)."""
    body = await request.json()
    model_name = body.get("name", "")

    if model_name == state.model_name:
        state._release_device()
        return {"status": "success"}

    raise HTTPException(status_code=404, detail="Model not found")


@app.get("/api/version")
async def version():
    """Get version info (Ollama /api/version endpoint)."""
    return {"version": "hailo-gateway-1.0.0"}


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import os

    host = os.environ.get("HAILO_GATEWAY_HOST", "0.0.0.0")
    port = int(os.environ.get("HAILO_GATEWAY_PORT", "11434"))

    print(f"Starting Hailo Ollama Gateway on {host}:{port}")
    print(f"Hailo platform available: {HAILO_AVAILABLE}")

    uvicorn.run(app, host=host, port=port)
