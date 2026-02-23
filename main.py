"""
LinkAI OpenAI Compatible API Proxy
将 Link-AI API 包装成 OpenAI 格式，支持多模型多 APP Code 映射。
"""

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any, Optional, Union

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from starlette.responses import StreamingResponse

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("linkai_proxy")


# ===== 配置 =====
LINKAI_API_KEY = os.getenv("LINKAI_API_KEY")
LINKAI_BASE_URL = os.getenv("LINKAI_BASE_URL", "https://api.link-ai.tech/v1").rstrip("/")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_HOST = os.getenv("LOCAL_HOST", "0.0.0.0")
LOCAL_PORT = int(os.getenv("LOCAL_PORT", "8000"))


def parse_model_app_mapping() -> dict[str, str]:
    """
    解析环境变量中的模型映射。

    支持两种格式：
    1) MODEL_MAPPINGS=gpt-4o:CM5Ex4OE,gpt-5-mini:jPzuwqKZ
    2) MODEL_MAP_gpt-4o=CM5Ex4OE
    """
    mappings: dict[str, str] = {}

    model_mappings = os.getenv("MODEL_MAPPINGS", "")
    if model_mappings:
        for pair in model_mappings.split(","):
            if ":" in pair:
                model_id, app_code = pair.split(":", 1)
                model_id = model_id.strip()
                app_code = app_code.strip()
                if model_id and app_code:
                    mappings[model_id] = app_code

    for key, value in os.environ.items():
        if key.startswith("MODEL_MAP_"):
            model_id = key[10:]
            app_code = value.strip()
            if model_id and app_code:
                mappings[model_id] = app_code

    return mappings


def parse_app_code_api_keys() -> dict[str, str]:
    """
    解析 APP Code -> API Key 映射。

    支持两种格式：
    1) APP_CODE_KEYS=CM5Ex4OE:key1,jPzuwqKZ:key2
    2) APP_CODE_CM5Ex4OE=key1
    """
    key_mappings: dict[str, str] = {}

    app_code_keys = os.getenv("APP_CODE_KEYS", "")
    if app_code_keys:
        for pair in app_code_keys.split(","):
            if ":" in pair:
                app_code, api_key = pair.split(":", 1)
                app_code = app_code.strip()
                api_key = api_key.strip()
                if app_code and api_key:
                    key_mappings[app_code] = api_key

    for key, value in os.environ.items():
        if key.startswith("APP_CODE_") and key != "APP_CODE_KEYS":
            app_code = key[9:]
            api_key = value.strip()
            if app_code and api_key:
                key_mappings[app_code] = api_key

    return key_mappings


MODEL_APP_MAPPINGS = parse_model_app_mapping()
APP_CODE_API_KEYS = parse_app_code_api_keys()
DEFAULT_APP_CODE = os.getenv("LINKAI_APP_CODE", "").strip()


def validate_config() -> None:
    if not LOCAL_API_KEY:
        raise RuntimeError("LOCAL_API_KEY is required")

    has_any_upstream_key = bool(LINKAI_API_KEY) or bool(APP_CODE_API_KEYS)
    if not has_any_upstream_key:
        raise RuntimeError("Either LINKAI_API_KEY or APP_CODE_* API keys must be configured")

    has_any_model_route = bool(DEFAULT_APP_CODE) or bool(MODEL_APP_MAPPINGS)
    if not has_any_model_route:
        raise RuntimeError("Configure LINKAI_APP_CODE or MODEL_MAPPINGS / MODEL_MAP_* first")


def mask_secret(value: Optional[str]) -> str:
    if not value:
        return "<empty>"
    return f"***{value[-4:]}"


logger.info("[LinkAI Proxy] Model mapping count: %d", len(MODEL_APP_MAPPINGS))
logger.info("[LinkAI Proxy] AppCode API key count: %d", len(APP_CODE_API_KEYS))
logger.info("[LinkAI Proxy] Default APP Code set: %s", "yes" if DEFAULT_APP_CODE else "no")
logger.info("[LinkAI Proxy] Global LinkAI API key: %s", mask_secret(LINKAI_API_KEY))


def get_app_code_for_model(model_id: Optional[str]) -> str:
    """根据模型 ID 获取对应的 APP Code。"""
    if not model_id:
        return DEFAULT_APP_CODE

    if model_id in MODEL_APP_MAPPINGS:
        return MODEL_APP_MAPPINGS[model_id]

    for mapping_model, app_code in MODEL_APP_MAPPINGS.items():
        if model_id.startswith(mapping_model):
            return app_code

    for mapping_model, app_code in MODEL_APP_MAPPINGS.items():
        if model_id.endswith(mapping_model):
            return app_code

    for mapping_model, app_code in MODEL_APP_MAPPINGS.items():
        if mapping_model in model_id:
            return app_code

    return DEFAULT_APP_CODE


def get_api_key_for_app_code(app_code: str) -> str:
    """根据 APP Code 获取对应的 API Key。"""
    if app_code in APP_CODE_API_KEYS:
        return APP_CODE_API_KEYS[app_code]
    if LINKAI_API_KEY:
        return LINKAI_API_KEY
    raise HTTPException(status_code=500, detail=f"No API key configured for app_code '{app_code}'")


def get_error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return err.get("message") or "LinkAI API Error"
            if isinstance(err, str):
                return err
        return "LinkAI API Error"
    except Exception:
        text = response.text.strip()
        return text[:300] if text else "LinkAI API Error"


def usage_payload(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _client

    validate_config()

    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    logger.info("[LinkAI Proxy] HTTP client initialized")

    try:
        yield
    finally:
        if _client is not None:
            await _client.aclose()
            logger.info("[LinkAI Proxy] HTTP client closed")


app = FastAPI(
    title="LinkAI OpenAI Compatible API",
    description="将 Link-AI API 包装成 OpenAI 兼容格式，支持多模型映射",
    version="2.1.0",
    lifespan=lifespan,
)


def http_client() -> httpx.AsyncClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _client


# ===== 鉴权 =====
security = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(security)) -> str:
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = api_key[7:] if api_key.lower().startswith("bearer ") else api_key
    token = token.strip()

    if token != LOCAL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token


# ===== Pydantic Models =====
class Message(BaseModel):
    role: str
    content: Union[str, list]


class ChatRequest(BaseModel):
    messages: list[Message]
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[list[str]] = None


@app.get("/v1/models")
async def list_models(_: str = Depends(verify_api_key)):
    models: list[dict[str, Any]] = []

    for model_id in MODEL_APP_MAPPINGS.keys():
        models.append(
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "linkai",
            }
        )

    if DEFAULT_APP_CODE:
        models.append(
            {
                "id": "linkai-default",
                "object": "model",
                "created": 0,
                "owned_by": "linkai",
            }
        )

    return {"object": "list", "data": models}


def build_linkai_body(request: ChatRequest, app_code: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "app_code": app_code,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
    }

    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.n is not None:
        body["n"] = request.n
    if request.stop:
        body["stop"] = request.stop

    return body


async def linkai_stream_generator(
    client: httpx.AsyncClient,
    linkai_body: dict[str, Any],
    model_id: str,
    app_code: str,
):
    body = dict(linkai_body)
    body["stream"] = True
    linkai_api_key = get_api_key_for_app_code(app_code)
    stream_id = f"chatcmpl-{secrets.token_hex(8)}"

    try:
        async with client.stream(
            "POST",
            f"{LINKAI_BASE_URL}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {linkai_api_key}"},
        ) as response:
            if response.status_code != 200:
                error_msg = get_error_message(response)
                yield f"data: {json.dumps({'error': {'message': error_msg}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                raw_data = line[6:].strip()
                if raw_data == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                try:
                    linkai_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    continue

                source_choices = linkai_data.get("choices") or []
                openai_choices = []

                for idx, choice in enumerate(source_choices):
                    delta_src = choice.get("delta") or {}
                    delta: dict[str, Any] = {}
                    if "role" in delta_src:
                        delta["role"] = delta_src["role"]
                    if "content" in delta_src:
                        delta["content"] = delta_src["content"]
                    if "tool_calls" in delta_src:
                        delta["tool_calls"] = delta_src["tool_calls"]
                    if not delta:
                        delta["content"] = ""

                    openai_choices.append(
                        {
                            "index": idx,
                            "delta": delta,
                            "finish_reason": choice.get("finish_reason"),
                        }
                    )

                openai_data: dict[str, Any] = {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": openai_choices,
                }

                usage = linkai_data.get("usage")
                if isinstance(usage, dict):
                    openai_data["usage"] = usage_payload(usage)

                yield f"data: {json.dumps(openai_data, ensure_ascii=False)}\n\n"

    except httpx.HTTPError as exc:
        logger.exception("Stream request failed: %s", exc)
        yield f"data: {json.dumps({'error': {'message': str(exc)}}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatRequest,
    _: str = Depends(verify_api_key),
):
    app_code = get_app_code_for_model(request.model)
    if not app_code:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found in mappings and no default APP Code configured",
        )

    logger.info("[LinkAI Proxy] Model '%s' -> APP Code '%s'", request.model, app_code)

    linkai_body = build_linkai_body(request, app_code)
    client = http_client()

    if request.stream:
        generator = linkai_stream_generator(
            client=client,
            linkai_body=linkai_body,
            model_id=request.model or "linkai-default",
            app_code=app_code,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    linkai_api_key = get_api_key_for_app_code(app_code)
    try:
        response = await client.post(
            f"{LINKAI_BASE_URL}/chat/completions",
            json=linkai_body,
            headers={"Authorization": f"Bearer {linkai_api_key}"},
        )
    except httpx.HTTPError as exc:
        logger.exception("Upstream request failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=get_error_message(response))

    try:
        linkai_response = response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Invalid upstream JSON response") from exc

    choices = linkai_response.get("choices") or []
    openai_choices = []
    for idx, choice in enumerate(choices):
        openai_choices.append(
            {
                "index": idx,
                "message": {
                    "role": (choice.get("message") or {}).get("role", "assistant"),
                    "content": (choice.get("message") or {}).get("content", ""),
                },
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        )

    usage = linkai_response.get("usage") or {}

    return {
        "id": f"chatcmpl-{secrets.token_hex(8)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model or "linkai-default",
        "choices": openai_choices,
        "usage": usage_payload(usage if isinstance(usage, dict) else {}),
    }


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": "2.1.0",
        "mappings_count": len(MODEL_APP_MAPPINGS),
    }


@app.get("/")
async def root():
    return {
        "name": "LinkAI OpenAI Compatible API",
        "version": "2.1.0",
        "description": "支持多模型多 APP Code 映射",
        "endpoints": {
            "chat_completions": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "health": "GET /health",
        },
        "model_mappings": MODEL_APP_MAPPINGS,
        "app_code_api_keys": {k: "***" for k in APP_CODE_API_KEYS.keys()},
        "default_app_code": DEFAULT_APP_CODE,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=LOCAL_HOST, port=LOCAL_PORT)
