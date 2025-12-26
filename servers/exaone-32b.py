"""
Shared EXAONE-3.5-32B Server
범용 OpenAI 호환 API - 모든 프로젝트에서 공유 가능

Endpoints:
- POST /v1/chat/completions - OpenAI 호환 채팅 API
- GET /v1/models - 모델 목록
- GET /health - 상태 확인
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch

app = FastAPI(title="Shared LLM Server - EXAONE-3.5-32B")

# ========== 모델 로딩 ==========
print("Loading model...")
model_id = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    trust_remote_code=True,
    device_map="auto",
)

device = next(model.parameters()).device
print(f"Model loaded! Device: {device}")


# ========== Request/Response 모델 ==========
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "EXAONE-3.5-32B-Instruct"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False  # 스트리밍 미지원 (추후 추가 가능)


# ========== LLM 생성 함수 ==========
def generate_response(messages: list[dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )


# ========== API 엔드포인트 ==========
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": model_id,
        "device": str(device)
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "EXAONE-3.5-32B-Instruct",
                "object": "model",
                "owned_by": "LGAI-EXAONE"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI 호환 채팅 API"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    response_text = generate_response(messages, request.max_tokens, request.temperature)

    return {
        "id": "chatcmpl-shared",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": -1,  # 실제 토큰 카운트 미구현
            "completion_tokens": -1,
            "total_tokens": -1
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)
