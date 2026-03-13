# Progress

This tracker maps the original roadmap to the current repository state.

Last updated: 2026-03-05 (frontend + persona validation sync)

Status legend:

- `[x]` done
- `[~]` in progress / partial
- `[ ]` not started

## Phase 1 - Foundational VM-Based Architecture

### Goal

Working, observable chatbot system that can be deployed and demoed quickly.

### Status

- `[x]` FastAPI backend + chat API implemented
- `[x]` BeeAI RAG pipeline implemented
- `[x]` ChromaDB integration implemented
- `[x]` OpenLIT/OpenTelemetry instrumentation implemented (with graceful degradation)
- `[x]` Dockerfile + docker-compose setup present
- `[x]` Frontend chat UI present
- `[x]` Frontend landing/marketing page integrated into `frontend/index.html` with hero CTA opening the chatbot widget
- `[x]` Frontend logo path fixed to match committed asset (`frontend/static/linkedin_logo.png`)
- `[~]` Splunk + OTEL collector setup depends on external infra and env configuration
- `[~]` Production hardening (secrets management, strict CORS, auth) remains incomplete

### Sponsor Requirement Tracking (Persona/Instruction Compliance)

- `[x]` `backend/agent.py` persona updated to a fun budtender style with Washington compliance framing
- `[x]` System instructions updated to block medical/therapeutic claims and treatment/cure/prevent language
- `[x]` Prohibited term set added in prompt instructions (sleep, pain, anxiety, PTSD, etc.)
- `[x]` Redirect behavior added for prohibited asks (fallback to flavor/terpenes/potency + suggest licensed healthcare professional)
- `[x]` Initial manual validation passed: prohibited-topic prompts now return compliance-safe responses
- `[~]` Ongoing validation still required with broader edge-case prompt sets
- `[~]` Guardrail backup layer (post-response policy check) not yet implemented in code

## Phase 2 - SaaS Multi-Tenancy + Live Agent Escalation

### Goal

Move from per-customer clone model to tenant-aware shared application architecture.

### Live Agent Handoff

- `[x]` Session states and handoff workflow implemented
- `[x]` Customer/agent WebSocket channels implemented
- `[x]` Agent queue and acceptance flow implemented
- `[~]` Timeout handling and user-facing wait/timeout messaging improved (recent update)
- `[~]` Reliability hardening (durable queue, reconnect handling, failover) incomplete

### Repo Hygiene / Security Hygiene (supporting work)

- `[~]` Local vector DB artifacts are still tracked in git (`my_chroma_db`); cleanup/untracking and ignore rules are not complete
- `[~]` Additional hardening still needed (tenant auth, RBAC, durable storage, full isolation tests, and DB artifact hygiene)

### Multi-Tenancy Core

- `[ ]` Tenant identity model enforced across all requests
- `[ ]` Tenant-aware auth/RBAC middleware
- `[ ]` Tenant-scoped data model in persistent storage (PostgreSQL)
- `[ ]` Tenant-scoped chat session persistence
- `[ ]` Tenant-scoped retrieval guarantees in RAG pipeline
- `[ ]` Tenant-scoped telemetry policy enforcement

## Phase 3 - Kubernetes, Scale, and Enterprise Standardization

### Goal

Elastic, resilient, enterprise-grade multi-tenant platform.

### Status

- `[ ]` Kubernetes deployment manifests and autoscaling
- `[ ]` Redis/session infrastructure
- `[ ]` PostgreSQL-backed control plane
- `[ ]` Centralized auth service and SSO readiness
- `[ ]` CI/CD and infrastructure-as-code for multi-env operations

## Next High-Impact Steps

1. Add tenant identity propagation (`tenant_id`) to request models, session state, and telemetry attributes.
2. Introduce durable persistence (PostgreSQL) for chat sessions, handoff queues, and audit records.
3. Add auth/RBAC middleware and enforce tenant scoping before hitting business logic.
4. Refactor retrieval layer to enforce tenant-scoped collections/indexes by design.
5. Add integration tests that explicitly verify cross-tenant isolation failure cases.
