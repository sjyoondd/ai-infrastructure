# Shared AI Services

NVIDIA DIGITS (GB10)에서 실행되는 공유 AI 인프라입니다.
여러 프로젝트에서 동일한 LLM과 Embedding 서버를 공유할 수 있습니다.

## 아키텍처

```
shared-ai-services/           ← 공유 AI 인프라 (이 프로젝트)
├── llm (EXAONE-32B)         :30000  ← 모든 프로젝트 공유
└── embedding (BGE-M3)       :8080   ← 모든 프로젝트 공유

projects/
├── korail_tomorrowro/        ← 프로젝트 A
│   └── qdrant               :6333
├── chatbot/                  ← 프로젝트 B
│   └── qdrant               :6334
└── another-project/          ← 프로젝트 C
```

## 서비스

| 서비스 | 포트 | 설명 |
|--------|------|------|
| LLM (EXAONE-3.5-32B) | 30000 | OpenAI 호환 채팅 API |
| Embedding (BGE-M3) | 8080 | 다국어 임베딩 API |

## 시작하기

### 1. 환경 설정

```bash
cp .env.example .env
# .env 파일에 HF_TOKEN 설정
```

### 2. 서비스 시작

```bash
# 공유 AI 서비스 시작 (먼저 실행!)
cd ~/문서/shared-ai-services
docker compose up -d

# 로그 확인 (모델 로딩에 5-10분 소요)
docker compose logs -f llm
```

### 3. 다른 프로젝트에서 사용

```python
from openai import OpenAI

# 공유 LLM 사용
client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="EXAONE-3.5-32B-Instruct",
    messages=[{"role": "user", "content": "안녕하세요"}]
)
print(response.choices[0].message.content)
```

```python
import requests

# 공유 Embedding 사용
response = requests.post(
    "http://localhost:8080/embed",
    json={"texts": ["안녕하세요", "Hello"]}
)
embeddings = response.json()["embeddings"]
```

## API 엔드포인트

### LLM Server (포트 30000)

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 상태 확인 |
| `/v1/models` | GET | 모델 목록 |
| `/v1/chat/completions` | POST | OpenAI 호환 채팅 |

### Embedding Server (포트 8080)

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 상태 확인 |
| `/embed` | POST | 텍스트 임베딩 |
| `/v1/embeddings` | POST | OpenAI 호환 임베딩 |

## Docker Network

다른 프로젝트에서 연결하려면 `shared-ai-network`에 참여해야 합니다:

```yaml
# 다른 프로젝트의 docker-compose.yml
services:
  my-app:
    networks:
      - shared-ai-network

networks:
  shared-ai-network:
    external: true
```

## 메모리 사용량

- LLM (EXAONE-32B): ~77GB
- Embedding (BGE-M3): ~2GB
- 총 필요 메모리: ~80GB (DIGITS 119GB 충분)
