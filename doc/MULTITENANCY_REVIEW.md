# Multitenancy Review

This review consolidates:

- your multitenancy architecture diagram,
- `Multi-Tenant Research.pdf`,
- and `Phase plan.pdf`,

then compares that plan against what is currently implemented in this repository.

## Quick Assessment

Your architecture direction is strong and security-minded. The layered boundary model (routing, service, data, network, and observability) is the right approach for enterprise AI SaaS.

Main gap: current code is still mostly single-tenant and in-memory for live-agent state. The plan is sound; implementation guardrails must now be encoded directly into request handling, persistence, and retrieval.

## What Is Good In The Plan

- Clear tenant boundary defense layers (ingress, service, data, network, telemetry).
- Explicit requirement for tenant-tagged telemetry to avoid observability leakage.
- Practical phased migration strategy (fast MVP first, then hardening/scale).
- Realistic acknowledgement of ChromaDB multitenancy limitations and future DB alternatives.
- Inclusion of cost/noisy-neighbor and latency concerns early in architecture design.

## Gaps To Close In Current Codebase

1. **Tenant identity propagation is missing**
   - No strict `tenant_id` field enforced through chat requests, sessions, websocket channels, and agent assignment paths.

2. **Auth/RBAC layer is not yet present**
   - Current endpoints do not enforce tenant-scoped authentication/authorization.

3. **Durable control-plane storage is missing**
   - Live-agent sessions and queues are in-memory only, which risks data loss and weak auditability.

4. **RAG retrieval is not tenant-scoped by contract**
   - Current retrieval path does not demonstrate hard tenant partitioning constraints.

5. **Cross-tenant test coverage is missing**
   - No isolation tests proving tenant A can never read/route/observe tenant B data.

## Recommended Implementation Sequence (Phase 2 Execution)

### Step 1 - Define Tenant Contract

- Require `tenant_id` on all customer and agent API payloads.
- Attach `tenant_id` to session models and every stored message.
- Reject requests with missing/invalid tenant identity before business logic.

### Step 2 - Add Persistent Control Plane (PostgreSQL)

- Create durable tables for:
  - `tenant`
  - `chat_session`
  - `chat_message`
  - `queue_entry`
  - `human_agent`
- Ensure every tenant-owned record includes `tenant_id` and indexed access paths.

### Step 3 - Enforce Auth and RBAC

- Validate JWT per request.
- Map token claims to `tenant_id` and role.
- Deny any operation where token tenant and requested tenant mismatch.

### Step 4 - Tenant-Scoped RAG

- Partition vector retrieval by tenant (separate collection/database namespace or explicit tenant filter that cannot be bypassed).
- Add server-side assertions and fail-safe defaults that deny retrieval if tenant context is absent.

### Step 5 - Tenant-Safe WebSockets and Handoff

- Bind websocket connection context to tenant.
- Enforce tenant match when agents accept queue items.
- Prevent cross-tenant broadcast or queue inspection.

### Step 6 - Telemetry Policy Enforcement

- Stamp every trace/log/metric with:
  - `tenant_id`
  - `session_id`
  - `trace_id`
  - `agent_id` (if applicable)
- Add checks in code to fail closed if required telemetry attributes are missing.

### Step 7 - Isolation Test Suite

- Add integration tests for:
  - cross-tenant retrieval attempts,
  - queue visibility isolation,
  - websocket routing isolation,
  - telemetry tag completeness.

## Practical Notes From Your Diagram

- The split between chatbot container, dashboard container, and shared OTEL pipeline is good.
- Introducing PostgreSQL as shared but tenant-scoped control plane is correct for session durability and auditing.
- Keep reverse proxy routing strict and deterministic (host/domain to tenant mapping).
- Do not rely on frontend-provided tenant values; derive trusted tenant identity from auth context.

## Definition of Done For "Phase 2 Multi-Tenancy"

Phase 2 should be considered complete only when:

- every request path is tenant-authenticated and tenant-authorized,
- every tenant record is durably stored and query-scoped,
- retrieval is tenant-hardened by architecture (not convention),
- live-agent handoff is tenant-isolated and durable,
- observability data is tenant-tagged and access-controlled,
- and automated tests prove cross-tenant access is blocked.
