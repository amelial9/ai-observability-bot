Detailed Architecture & Data Flow
This document describes the detailed component interaction and data flow for the AI Observability Bot, including internal application modules and external observability integrations.

System Components
1. External Access Layer
User (Browser): End-user interacting with the chatbot widget.
Squarespace: Hosts the static frontend assets (HTML/CSS/JS).
Caddy (Reverse Proxy):
Ports: :80 (HTTP), :443 (HTTPS)
Role: Handles SSL termination and proxies API/WebSocket requests to the backend.
2. Application Layer (ai-observability-bot)
Running as a Docker container, exposing port :8001.

FastAPI Application (main.py):
Entry point for all HTTP and WebSocket requests.
Routes: /chat, /ws/customer/{id}, /ws/agent/{id}.
Session Manager (live_agent_system.py):
Manages session state (AI vs. Live Agent).
Handles "Handoff" logic (detecting keywords like "human", "agent").
Maintains waiting queues.
Connection Manager (websocket_manager.py):
Maintains active WebSocket connections for real-time bi-directional communication.
Broadcasts messages between Customer and Agent.
RAG Engine (agent.py):
BeeAI / LangChain: Orchestrates the retrieval and generation process.
ChromaDB: Embedded vector store for retrieving relevant FAQ context.
OpenAI Integration: Calls external LLM API for response generation.
OpenLIT / OTEL SDK:
Auto-instruments LLM calls and application traces.
Sends telemetry data to the OTEL Collector.
3. Observability Infrastructure
OTEL Collector (Container):
Ports: :4327 (gRPC), :4328 (HTTP)
Role: Aggregates traces, metrics, and logs. Process batches.
Splunk (OCI VM):
IP: 10.0.0.249
Ports: :8088 (HEC), :8000 (Web UI)
Role: Centralized log analysis and dashboarding.
Detailed Data Flow
Chat Flow (User -> Bot)
Initiation: User sends message from Browser.
Proxy: Caddy receives request on :443, forwards to ai-observability-bot:8001.
Handling: FastAPI receives request.
Session Check: SessionManager checks if user is in "Live Agent" mode.
If Live Agent: Message routed via ConnectionManager directly to Agent WebSocket.
If AI: Message passed to RAG Engine.
RAG Process:
agent.py queries ChromaDB for context.
Constructs prompt with context.
Calls OpenAI API.
Response: Generated answer returned to User.
Observability Flow (Bot -> Splunk)
Instrumentation: OpenLIT intercepts the OpenAI call and RAG execution.
Trace Generation: Spans are created for "Retrieval", "Generation", and "Total Request".
Export: SDK sends OTLP data to OTEL Collector on :4328 (HTTP).
Processing: Collector batches data (10s timeout or 1000 items).
Ingestion: Collector sends batch to Splunk HEC on :8088 (HTTPS).
Visualization: Data appears in Splunk Web UI on :8000.
Mermaid Diagram (Component Level)
graph TD
    subgraph "External"
        User[User Browser]
        SQ[Squarespace Host]
        OpenAI[OpenAI API]
    end

    subgraph "Ingress"
        Caddy[Caddy Proxy<br/>:80 / :443]
    end

    subgraph "App Container (:8001)"
        FastAPI[FastAPI App]
        Sess[Session Manager]
        WS[Connection Manager]
        RAG[RAG Engine]
        Chroma[(ChromaDB)]
        SDK[OpenLIT SDK]
    end

    subgraph "Observability"
        OTEL[OTEL Collector<br/>:4327 / :4328]
        Splunk[Splunk VM<br/>:8088 HEC]
    end

    User -->|HTTPS| SQ
    User -->|WSS/HTTPS| Caddy
    Caddy --> FastAPI

    FastAPI --> Sess
    FastAPI --> WS
    FastAPI --> RAG

    RAG --> Chroma
    RAG -->|Generate| OpenAI

    RAG -.->|Instrument| SDK
    SDK -.->|OTLP| OTEL
    OTEL -.->|HEC| Splunk

---

**Multi-tenancy:** For as-is critique, risks, and multi-tenant target architecture (security, observability, ChromaDB isolation, phases, checklist), see [MULTI_TENANT_ARCHITECTURE_PROPOSAL.md](./MULTI_TENANT_ARCHITECTURE_PROPOSAL.md).