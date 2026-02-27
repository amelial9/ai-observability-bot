# Progress

This tracker maps the original roadmap to the current repository state.

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
- `[~]` Splunk + OTEL collector setup depends on external infra and env configuration
- `[~]` Production hardening (secrets management, strict CORS, auth) remains incomplete

## Phase 2 - SaaS Multi-Tenancy + Live Agent Escalation

### Goal

Move from per-customer clone model to tenant-aware shared application architecture.

### Live Agent Handoff

- `[x]` Session states and handoff workflow implemented
- `[x]` Customer/agent WebSocket channels implemented
- `[x]` Agent queue and acceptance flow implemented
- `[~]` Reliability hardening (durable queue, reconnect handling, failover) incomplete

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
