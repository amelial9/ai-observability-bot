# Project Overview

## What This Project Is

`ai-observability-bot` is a Python/FastAPI customer support chatbot system with:

- RAG-based FAQ answering using BeeAI + ChromaDB + OpenAI,
- live-agent handoff over WebSockets,
- and AI observability via OpenLIT + OpenTelemetry, with Splunk as downstream analytics.

## Current Runtime Architecture (as implemented)

- **Backend API:** FastAPI app in `backend/main.py`
- **AI Orchestration:** BeeAI workflow in `backend/agent.py`
- **Vector Store:** ChromaDB persistent collection (`backend/agent.py`, `backend/extraction.py`)
- **Live Agent:** In-memory session/queue manager in `backend/live_agent_system.py` + WebSocket routing in `backend/websocket_manager.py`
- **Frontend:** Static web UI under `frontend/`
- **Deployment:** Dockerfile + `docker-compose.yml` (single application container; OTEL collector expected externally)

## Core Tooling

- **Primary AI framework:** BeeAI (`beeai-framework`)
- **Primary web framework:** FastAPI + Uvicorn
- **Observability:** OpenLIT + OpenTelemetry OTLP HTTP exporter
- **LLM provider:** OpenAI
- **Data/RAG libs:** ChromaDB, sentence-transformers, pandas/openpyxl

## Important Current-State Notes

- Multi-tenancy is not fully implemented in code yet.
- Live-agent state is currently in memory (not durable; reset on restart).
- No tenant-aware auth/RBAC layer is enforced yet.
- Chroma usage appears single-tenant in current code paths.
- `README.md` and code use slightly different Chroma paths/names in places; align later as part of hardening.

## Intended Direction

Based on planning docs and architecture drafts, the roadmap is:

1. Phase 1: working single-tenant/single-environment system with observability.
2. Phase 2: tenant-aware SaaS architecture + live-agent maturity.
3. Phase 3: Kubernetes-scale, enterprise-grade operational model.

See `doc/PROGRESS.md` and `doc/MULTITENANCY_REVIEW.md` for detailed status.
