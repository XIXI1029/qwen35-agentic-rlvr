# -*- coding: utf-8 -*-
# =====================================================================
# serve.py —— 部署：OpenAI 兼容的本地 API
#
# 背景vLLM 官方不支持 Windows，所以这里用纯 transformers 起一个
#   /v1/chat/completions 端点，协议与 OpenAI 一致：
#     curl http://localhost:8000/v1/chat/completions \
#          -H "Content-Type: application/json" \
#          -d '{"model":"local","messages":[{"role":"user","content":"15*17+3=?"}]}'
#   客户端 openai.OpenAI(base_url="http://localhost:8000/v1", api_key="x") 即可调用。
#
# 用法 python scripts/serve.py --model outputs/qwen3.5-4b-grpo --port 8000
# =====================================================================

from __future__ import annotations

import argparse
import json

import torch
from _bootstrap import logger

# 延迟 import
from transformers import AutoModelForImageTextToText, AutoTokenizer


def build_app(model_path: str, max_new: int = 256):
    """加载模型并返回一个"处理单条请求"的闭包，供 HTTP 层调用。"""
    from fastapi import FastAPI, Request

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=False,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    logger.info(f"模型已加载: {model_path}")

    app = FastAPI(title="Qwen3.5 Agent API")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "local", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        msgs = body.get("messages", [])
        # 只保留 role/content（本 demo 不支持工具）
        content = msgs[-1]["content"] if msgs else ""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(gen, skip_special_tokens=True)
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant",
                                                 "content": reply},
                         "finish_reason": "stop"}],
        }
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/qwen3.5-4b-grpo")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    app = build_app(args.model)
    logger.info(f"启动 OpenAI 兼容服务: http://127.0.0.1:{args.port}/v1")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
