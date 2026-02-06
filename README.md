# Hailo Ollama Gateway

An Ollama-compatible REST API gateway for Hailo AI accelerators.

## Overview

This gateway translates Ollama REST API calls to Hailo's native RPC protocol,
allowing you to use Hailo AI accelerators with any Ollama-compatible client.

## Architecture

```
┌─────────────┐     HTTP      ┌──────────────────┐    HRPC    ┌──────────────┐
│   Client    │ ───────────▶  │  FastAPI Gateway │ ────────▶  │ HailoRT      │
│  (curl,     │  /api/chat    │  (Port 11434)    │   Binary   │ Server       │
│   OpenWebUI │               │                  │   Proto    │ (Port 12133) │
│   etc.)     │ ◀───────────  │                  │ ◀────────  │              │
└─────────────┘    NDJSON     └──────────────────┘            └──────────────┘
                  Streaming
```

## Prerequisites

- HailoRT installed and running
- Python 3.8+
- Hailo platform Python bindings (`hailo_platform`)

## Installation

```bash
cd /home/jpop/devel/ollama_gateway
pip install -r requirements.txt
```

## Usage

### Direct Start

```bash
# Set your HEF model path
export HAILO_HEF_PATH=/path/to/your/llm.hef

# Start the gateway
python hailo_ollama_gateway.py
```

### With Nginx (Production)

1. Install the systemd service:
```bash
sudo cp hailo-ollama-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hailo-ollama-gateway
sudo systemctl start hailo-ollama-gateway
```

2. Configure Nginx:
```bash
sudo cp nginx.conf /etc/nginx/sites-available/hailo-ollama
sudo ln -s /etc/nginx/sites-available/hailo-ollama /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET/HEAD | Health check |
| `/api/generate` | POST | Generate text (streaming/non-streaming) |
| `/api/chat` | POST | Chat completion (streaming/non-streaming) |
| `/api/tags` | GET | List available models |
| `/api/ps` | GET | List running models |
| `/api/pull` | POST | Load a HEF model |
| `/api/delete` | DELETE | Unload a model |
| `/api/version` | GET | Version info |

## Examples

### Generate Text

```bash
curl -H "Content-Type: application/json" http://localhost:11434/api/generate -d '{
  "model": "hailo-llm",
  "prompt": "What is machine learning?",
  "stream": false
}'
```

### Streaming Chat

```bash
curl -H "Content-Type: application/json" http://localhost:11434/api/chat -d '{
  "model": "hailo-llm",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ]
}'
```

### Load a Model

```bash
curl -H "Content-Type: application/json" http://localhost:11434/api/pull -d '{
  "name": "/path/to/your/model.hef"
}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HAILO_HEF_PATH` | "" | Path to HEF model to load on startup |
| `HAILO_GATEWAY_HOST` | "0.0.0.0" | Host to bind to |
| `HAILO_GATEWAY_PORT` | "11434" | Port (matches Ollama default) |

## Compatibility

This gateway is designed to be compatible with:
- OpenWebUI
- LangChain (Ollama provider)
- Ollama CLI
- Any Ollama-compatible client

## Limitations

- **Embeddings**: Not supported (Hailo LLM doesn't expose embeddings directly)
- **Model Registry**: No remote model pulling - provide local HEF paths
- **Vision**: VLM support requires the VLM model to be loaded separately
