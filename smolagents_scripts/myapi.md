## vLLM Server (Windows + Docker)

Run Qwen3 models with vLLM either via a local PowerShell virtual environment or Docker with NVIDIA GPUs.

Notes:
- vLLM is most reliable on Linux. On Windows, Docker with GPU is recommended.
- A PowerShell venv flow is provided; if native install fails, use Docker.

### Prerequisites
- NVIDIA GPU + latest drivers
- NVIDIA Container Toolkit (for Docker GPU)
- PowerShell 7+
- Python 3.10 or 3.11 (for venv)

### Configuration (.env)
Create a `.env` in this folder (values shown with defaults used by scripts):

```
# Model and server
MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
PORT=8000
MAX_MODEL_LEN=28672

# Performance
GPU_MEMORY_UTILIZATION=0.70
KV_CACHE_DTYPE=auto

# Reasoning (Thinking models)
ENABLE_REASONING=true
# deepseek_r1 is used by default for "Thinking" models; otherwise scripts default to qwen3
REASONING_PARSER=deepseek_r1

# Optional: Hugging Face cache directory on host (used by Docker)
# HF_HOME=C:\\Users\\<you>\\.cache\\huggingface
```

Key behavior:
- PowerShell server: `ENABLE_REASONING=true` by default. If `MODEL_ID` contains "Thinking", it defaults to `REASONING_PARSER=deepseek_r1`; otherwise it defaults to `REASONING_PARSER=qwen3`.
- Docker image: defaults to `ENABLE_REASONING=true` and `REASONING_PARSER=deepseek_r1`.

### Option B: Docker (recommended on Windows)
Build and run with GPU (builds the image, mounts HF cache, prints logs):

```
pwsh -File .\run_docker.ps1
```

The server will be available at `http://localhost:8000/v1` (OpenAI-compatible).

Docker details:
- Base image: CUDA 12.9 + cuDNN runtime (Ubuntu 22.04)
- Installs GPU PyTorch (CUDA 12.9) and vLLM in one layer
- Mounts host HF cache to `/root/.cache/huggingface`
- `--shm-size=2g` for stability
- Honors `.env` for all variables above

Compose alternative:

```
docker compose up --build -d
docker compose logs -f
```

### Endpoints and quick tests
- Models list:

```
curl http://localhost:8000/v1/models
```

- Simple chat completion (PowerShell script does this end-to-end):

```
pwsh -File .\test_server.ps1
```

### Notes for Qwen3 Thinking models
- Keep `ENABLE_REASONING=true` and set `REASONING_PARSER=deepseek_r1` (default in Docker and applied by scripts when appropriate).
- For non-thinking models, either set `ENABLE_REASONING=false` or set `REASONING_PARSER=qwen3` if enabling reasoning.

### Where things live
- `setup_venv.ps1`: Creates venv, installs CUDA 12.9 PyTorch and vLLM.
- `start_server.ps1`: Starts vLLM; reads `.env`; applies reasoning flags.
- `test_server.ps1`: Simple OpenAI-compatible test (chat + prints reasoning if present).
- `print_vram.py`: VRAM/KV cache estimator (also baked into the Docker image and runs at container start).
- `Dockerfile`: Builds a self-contained vLLM image; runs `print_vram.py` then `vllm serve`.
- `docker-compose.yml`: Optional compose setup mirroring `run_docker.ps1`.

### API: OpenAI-compatible interface
- **Protocol**: HTTP+JSON; streaming via Server-Sent Events (SSE)
- **Base URL**: `http://localhost:8000/v1`
- **Endpoints**:
  - `GET /v1/models` — list available models
  - `POST /v1/chat/completions` — chat API
  - `POST /v1/completions` — legacy text completion API
  - Auth: none by default (no API key required)

#### Chat Completions request
Body fields (common ones):
- **model** (string, required): e.g., `Qwen/Qwen3-4B-Thinking-2507`
- **messages** (array, required): `[{ role: system|user|assistant, content: string }]`
- **max_tokens** (int): cap on new tokens
- **temperature** (float): randomness, 0–2
- **top_p** (float): nucleus sampling, 0–1
- **stream** (bool): enable SSE streaming
- **stop** (string | string[]): stop strings
- Additional generation controls supported by vLLM (optional): `top_k`, `min_p`, `repetition_penalty`, `length_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `n`

Example request:

```json
{
  "model": "Qwen/Qwen3-4B-Thinking-2507",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Say hello in one short sentence." }
  ],
  "temperature": 0.7,
  "max_tokens": 256
}
```

#### Chat Completions response (non-streaming)
Key fields:
- **id**, **object**, **created**, **model**
- **choices[0].message.content**: final answer text
- **choices[0].message.reasoning_content**: reasoning text (present for Thinking models)
- **choices[0].finish_reason**: `stop` | `length` | ...
- **usage**: `prompt_tokens`, `completion_tokens`, `total_tokens`

Response snippet:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1731000000,
  "model": "Qwen/Qwen3-4B-Thinking-2507",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "(hidden chain-of-thought or structured reasoning)",
        "content": "Hello!"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15 }
}
```

#### Streaming
Set `"stream": true` to receive SSE chunks until a `data: [DONE]` sentinel. Each chunk contains a delta like OpenAI:

```text
data: { "id": "...", "object": "chat.completion.chunk", "choices": [{ "delta": { "content": "He" } }] }
data: { "id": "...", "object": "chat.completion.chunk", "choices": [{ "delta": { "content": "llo" } }] }
data: [DONE]
```

Example streaming with curl:

```bash
curl -N -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-4B-Thinking-2507",
  "messages": [
    {"role":"user","content":"Stream one short sentence."}
  ],
  "stream": true
}' http://localhost:8000/v1/chat/completions
```

#### Legacy Completions request
Body fields (common ones):
- **model** (string, required)
- **prompt** (string | string[])
- **max_tokens**, **temperature**, **top_p**, **stream**, **stop**, and the same optional controls as above

Example request:

```json
{
  "model": "Qwen/Qwen3-4B-Thinking-2507",
  "prompt": "Say hello in one short sentence.",
  "max_tokens": 128,
  "temperature": 0.7
}
```

#### Response fields (summary)
- Top-level: `id`, `object`, `created`, `model`
- For chat: `choices[].message.role`, `choices[].message.content`, `choices[].message.reasoning_content` (Thinking models), `choices[].finish_reason`
- For legacy completions: `choices[].text`, `choices[].finish_reason`
- Usage accounting: `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`

#### Models list

```bash
curl http://localhost:8000/v1/models
```

