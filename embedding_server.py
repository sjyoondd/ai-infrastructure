"""
Shared BGE-M3 Embedding Server
다국어/한국어 지원 임베딩 서버 - 모든 프로젝트에서 공유 가능

Endpoints:
- POST /embed - 텍스트 임베딩
- POST /v1/embeddings - OpenAI 호환 임베딩 API
- GET /health - 상태 확인
"""
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch

app = FastAPI(title="Shared Embedding Server - BGE-M3")

# ========== 모델 로딩 ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Memory optimization: load with half precision on GPU
if device == "cuda":
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    model.half()  # FP16 reduces VRAM by ~50%
else:
    model = SentenceTransformer('BAAI/bge-m3', device=device)
print(f"BGE-M3 model loaded on {device}")


# ========== Request/Response 모델 ==========
class EmbedRequest(BaseModel):
    texts: list[str]

class OpenAIEmbedRequest(BaseModel):
    input: str | list[str]
    model: str = "bge-m3"


# ========== API 엔드포인트 ==========
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "BAAI/bge-m3",
        "device": device
    }


@app.post("/embed")
async def embed(request: EmbedRequest):
    """기본 임베딩 엔드포인트"""
    with torch.no_grad():  # Disable gradient tracking to save memory
        embeddings = model.encode(
            request.texts,
            device=device,
            normalize_embeddings=True,
            batch_size=32  # Process in batches to limit peak memory
        ).tolist()
    return {
        "embeddings": embeddings,
        "model": "bge-m3",
        "dimensions": len(embeddings[0]) if embeddings else 0
    }


@app.post("/v1/embeddings")
async def openai_embed(request: OpenAIEmbedRequest):
    """OpenAI 호환 임베딩 API"""
    texts = request.input if isinstance(request.input, list) else [request.input]
    with torch.no_grad():
        embeddings = model.encode(
            texts,
            device=device,
            normalize_embeddings=True,
            batch_size=32
        ).tolist()

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i
            }
            for i, emb in enumerate(embeddings)
        ],
        "model": request.model,
        "usage": {
            "prompt_tokens": -1,
            "total_tokens": -1
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
