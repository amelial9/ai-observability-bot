# Multi-Tenant Architecture: As-Is Critique & Target Proposal

**Document type:** Architecture critique + multi-tenant target proposal
**Focus:** Security + Observability first, then scalability and cost
**Context:** Customer service AI agent for target retailer; next phase = MULTI-TENANCY with strong isolation.

---

## As-Is Architecture Summary

| Component | Current state |
|-----------|----------------|
| **Chatbot (VM-Chatbot / container)** | Single container: FastAPI (Human Agent Dashboard + API + agent orchestrator), RAG/BeeAI, OpenAI calls, ChromaDB embedded. Port 8001. |
| **OTEL** |Yes: OpenLIT + OTEL SDK in app; exporter to OTEL Collector (HTTP 4328). Collector: 0.0.0.0:4327/4328, no auth. |
| **Exporter target** | Yes: Splunk on VM-B via HEC (8088). |
| **Auth (customer)** | None. Session: client-supplied `session_id` (no signing/validation). |
| **Auth (agent dashboard)** | None. `/api/agent/login` accepts any `agent_id` + `name`; no JWT/OIDC/API key. |
| **Data store** | ChromaDB on local disk (`my_chroma_db`), single collection `company_faqs`. No tenant concept. |
| **Secrets** | Env vars (`.env`): OPENAI_API_KEY, SENDER_PASSWORD, etc. HEC token **hardcoded** in `otel-splunk-4-5.yaml`. |
| **Network** | Caddy :80/:443 → app :8001. Outbound: OpenAI, OTEL collector. OTLP listener bound to all interfaces. |

---

# A) As-Is Architecture Critique

## 1. User connection / session issues

- **Client-supplied `session_id`**: Callers can send any UUID; no binding to tenant, user, or device. Attacker can guess or enumerate session IDs and read/impersonate conversations.
- **No session binding**: Sessions are not tied to tenant, so multi-tenant would mix tenants in the same in-memory session store.
- **CORS `allow_origins=["*"]`**: Any origin can call the API (needed today for Squarespace embed but unacceptable for multi-tenant B2B without additional auth and origin validation).

## 2. AuthN / AuthZ gaps

- **No customer auth**: `/chat`, `/ws/customer/{session_id}` have no authentication. Cannot attribute requests to a tenant or user.
- **No agent auth**: `/api/agent/login`, `/api/agent/accept/{session_id}`, `/ws/agent/{agent_id}` are unauthenticated. Anyone can register as agent, accept any session, and read all conversations.
- **No tenant-scoped authZ**: Even with future authN, there is no tenant_id in requests or authorization checks; every endpoint would allow cross-tenant access if added naively.

## 3. Tenant isolation failures

- **Single ChromaDB collection**: One `company_faqs` collection for all data. Multi-tenant would leak one tenant’s FAQs to another via shared retrieval.
- **Single in-memory session store**: `SessionManager.sessions` is global; no tenant_id on sessions. All tenants’ sessions would be in one namespace.
- **Single queue**: `waiting_queue` is global; handoff queue would mix tenants.
- **No tenant in traces/logs**: Spans and logs do not carry tenant_id; observability cannot be scoped per tenant and could leak tenant context in shared dashboards.

## 4. ChromaDB isolation

- **One collection, one path**: No per-tenant collection or namespace. Any bug or misconfiguration exposes all FAQ data to all callers.
- **No RLS or metadata filter**: Queries do not filter by tenant; adding tenant later without a strict pattern risks bypasses.
- **Embedded in app container**: Same process as API; compromise of app = full access to DB. No separate security boundary.

## 5. Data-at-rest and key management

- **ChromaDB on host volume**: Data at rest not encrypted (filesystem-level encryption not assumed). No key rotation for DB.
- **HEC token in config**: `otel-splunk-4-5.yaml` contains Splunk HEC token in plaintext; in repo or config management this is a secret leak.
- **OpenAI API key in env**: Standard but single key for all tenants; no per-tenant keys or quota isolation at key level.
- **No key rotation story**: No process for rotating API keys, HEC token, or DB credentials.

## 6. PII handling

- **Full prompts/responses in spans**: `agent.py` sets `llm.prompt` and `llm.response` as span attributes; these can contain PII and are sent to Splunk. No redaction.
- **Logs**: `print()` of user queries and errors can include PII; no structured logging with redaction or tenant-aware log routing.
- **Session messages in memory**: Full conversation history in `SessionInfo.messages` with no retention or encryption; export to Splunk or logs would expose PII.

## 7. Prompt injection / LLM abuse

- **User content straight to LLM**: User input is concatenated with retrieved FAQ context and sent to OpenAI; no strict prompt boundaries or injection controls.
- **No per-tenant guardrails**: OpenLIT guards use a single config; no tenant-specific policies or blocklists.
- **No rate limiting**: A single abusive tenant can exhaust OpenAI quota and affect others.

## 8. SSRF and outbound security

- **OTEL endpoint from env**: If endpoint is user-controllable (e.g. misconfiguration), app could be used as SSRF to internal OTLP. Currently hardcoded in agent.py as fallback.
- **OpenAI only**: No allowlist of outbound hosts; dependency on env and DNS for correct endpoint.

## 9. Rate limiting and abuse

- **No rate limiting**: No per-IP, per-session, or per-tenant limits on `/chat` or WebSockets. Enables DoS and cost abuse.
- **No throttling on agent APIs**: Agent endpoints can be spammed (e.g. queue polling, accept/end).

## 10. Audit logs

- **No audit trail**: No logging of who (tenant/user/agent) did what (login, accept, end session, access queue). Cannot prove compliance or investigate incidents.
- **Splunk**: Has data but no structured “audit” events for auth and sensitive actions.

## 11. OTLP endpoint exposure

- **Collector binds 0.0.0.0:4327/4328**: Any host that can reach the collector can push traces/metrics/logs. No auth on OTLP; fake or malicious telemetry can be injected.
- **No tenant in OTLP**: Exporters do not add tenant_id; cannot attribute or filter telemetry by tenant; risk of cross-tenant visibility in Splunk.

## 12. Splunk ingestion security

- **HEC token in repo**: Token in YAML and README; if repo is shared or deployed broadly, token is compromised.
- **TLS `insecure_skip_verify: true`**: HEC connection does not verify Splunk certificate; MITM possible on the link.
- **No HEC acknowledgment**: Reliance on Splunk’s default behavior; no explicit ack or retry policy documented for at-least-once delivery.

## 13. VM / container hardening

- **Monolithic container**: Web, API, agent logic, ChromaDB, and SDK in one process. Blast radius is maximum; no principle of least privilege per component.
- **Agent Dashboard and chatbot in same container**: Customer-facing UI and backend are in the same service; compromise of one exposes the other; scaling is tied.
- **Single point of failure**: One app container, one ChromaDB instance, one collector; no redundancy.
- **No read-only filesystem**: Container not run read-only; no explicit security hardening (e.g. non-root, drop caps) visible in compose.

## 14. Dependency risks

- **OpenAI API dependency**: Single external LLM; outage or policy change affects all tenants.
- **ChromaDB embedded**: Upgrades and backups are tied to app lifecycle; no managed DB with backups and point-in-time recovery.

## 15. Architecture hygiene

- **UI and backend together**: 客服窗口 (customer UI) and chatbot API in same container: mixing public-facing and internal boundaries; recommend separating (e.g. static hosting + API-only container).
- **Agent dashboard same host**: Agent dashboard served from same app as customer API; agent endpoints should be in a separate network segment and require strong auth.
- **Scaling limits**: Single replica; session state in memory; ChromaDB on local disk—cannot horizontally scale without stateful design change.

---

# B) Multi-Tenant Target Architecture Proposal

## Tenant isolation strategy: **Option 1 (shared app + per-tenant data + strict authZ)** with elements of Option 2

**Choice:** Shared application tier with **per-tenant ChromaDB collections** (or per-tenant metadata filter if using a single DB with strict metadata), **tenant-scoped session store**, and **strict authZ at every layer**. Option 3 (per-tenant deployments) is not recommended for MVP: higher cost and operational load; we can revisit for “vault” tenants later.

**Rationale:**

- Single codebase and one deployment to operate; faster to ship MVP multi-tenant.
- Isolation via tenant_id in every request (JWT or API key), enforced in API, session store, ChromaDB, and observability.
- Per-tenant collections (or equivalent) give clear data boundaries and simpler compliance story.
- Option 2 (row-level only) in ChromaDB is possible but ChromaDB’s primary isolation is by collection; per-tenant collections are clearer and avoid RLS bugs.
- Option 3 reserved for tenants that require physical separation (e.g. regulated verticals).

## ChromaDB placement and tenant isolation

- **Recommendation:** Move ChromaDB **out of the app container** to a **dedicated ChromaDB service** (or managed vector DB, e.g. Pinecone/Weaviate with tenant metadata) in the same VPC.
- **Isolation:** One **collection per tenant** (e.g. `faqs_<tenant_id>`). App resolves tenant from auth, then uses only that collection. No cross-tenant query possible if authZ is correct.
- **Alternative:** Single ChromaDB with metadata filter `tenant_id = X` on every query; enforce in a single data access layer and tests. Prefer per-tenant collections for simplicity and auditability.
- **Backup & encryption:** Use volume or DB-level encryption; automated backups; test restore.

## Container and service boundaries

- **Separate containers/services:**
  - **Customer-facing API** (FastAPI): `/`, `/chat`, `/ws/customer/*`, `/static`. No ChromaDB in process; calls internal RAG/agent service.
  - **Agent API** (or same app with strict path-based routing): `/agent-dashboard`, `/api/agent/*`, `/ws/agent/*`. Behind internal LB or separate ingress with IP/auth restrict.
  - **RAG/Agent worker**: Loads embedding model, talks to ChromaDB and OpenAI. No direct user traffic; called by API. Can scale independently.
  - **ChromaDB**: Dedicated container or managed service; network restricted to RAG/worker and backup.
  - **OTEL Collector**: Dedicated; receives only from app/worker; no user traffic.
- **Same container only where justified:** API + agent dashboard can stay in one process initially if path-based routing and authZ are strict, and agent routes are not exposed on the same URL as customer chat. Prefer splitting agent dashboard to separate service or at least separate auth domain.

## Network segmentation

- **VPC:** One VPC; separate subnets: public (LB only), app (API + worker), data (ChromaDB, Splunk), observability (collector).
- **Security groups:**
  - LB: 443 from internet; forward to API only.
  - API: from LB only; outbound to Worker, Collector (and OpenAI).
  - Worker: from API only; outbound to ChromaDB, OpenAI, Collector.
  - ChromaDB: from Worker (and backup host) only.
  - Collector: from API and Worker only (e.g. 10.0.0.0/24 or specific security group); **not** 0.0.0.0.
- **Private OTLP:** Collector listens on internal IP only; no public IP. API and Worker use internal endpoint (e.g. `http://otel-collector:4318`).
- **TLS:** TLS everywhere: customer → LB, LB → API, API → Worker (internal), Worker → ChromaDB if supported, Collector → Splunk HEC. Disable `insecure_skip_verify`; use proper certs or internal CA.
- **mTLS (optional):** For Collector ↔ Splunk or API ↔ Worker, mTLS adds strong identity; recommend for Phase 2.

## Secrets management

- **Vault or cloud native:** Use HashiCorp Vault, or OCI Secrets / AWS Secrets Manager / Azure Key Vault. No secrets in YAML or README.
- **Rotation:** HEC token, OpenAI key, DB credentials rotatable without app restart (e.g. short TTL + refresh, or restart with new secret). Document rotation runbook.
- **App:** Read secrets at startup or via sidecar; env vars from secret store, not from checked-in files.

## Observability design

- **Trace IDs:** Preserve W3C TraceContext; ensure tenant_id in resource or span attributes on every span.
- **Span naming:** Consistent convention, e.g. `faq_agent.retrieve`, `faq_agent.llm`, `chat.request`; include tenant_id in resource attributes, not in span name (to avoid high cardinality).
- **Baggage / tenant propagation:** Add `tenant_id` to OTel baggage (and optionally header) at API edge; propagate through all services and into logs. Do not put PII in baggage.
- **Log correlation:** Structured logging (JSON); include trace_id, span_id, tenant_id in every log line; no PII in logs by default.
- **Redaction:** Redact or hash PII in spans (e.g. do not set full `llm.prompt`/`llm.response`; use length and hash only, or redact in collector). Apply same policy in logs.
- **Sampling:** Head-based or tail-based sampling per tenant if needed; sample errors and slow requests at 100% for SLO.
- **Dashboards & SLOs:** Per-tenant (or tenant-tier) dashboards; SLOs on latency (p95), error rate, and optionally token usage; alerts by tenant.

## Security controls

- **WAF:** In front of LB; block known bad patterns, limit payload size.
- **Bot protection:** CAPTCHA or similar on chat entry for unauthenticated flows; for B2B, prefer API key + rate limit.
- **Throttling:** Per-tenant and per-user rate limits on `/chat` and WebSockets; per-IP on login if applicable.
- **OTLP:** Collector allowed only from API/Worker security group (e.g. 10.0.0.0/24 or SG); no public OTLP.
- **Splunk RBAC:** Roles per tenant or per team; index-level or field-level access so tenant A cannot see tenant B’s data.
- **Audit trails:** Log all agent logins, session accept/end, and admin actions to a dedicated audit index with immutable retention.

---

# C) Output Format

## 1) System diagram (trust boundaries and data flows)

```
+------------------+     HTTPS      +------------------+
|   User Browser   |----------------|  Caddy / LB      |
|  (Squarespace    |                |  :443            |
|   embed)         |                +--------+---------+
+------------------+                         |
         |                                   | (trust: internet -> edge)
         |                                   v
         |                         +------------------+
         |                         |  API Container   |
         +------------------------>|  (FastAPI)      |
         |  /chat, /ws/customer/*  |  - AuthN/AuthZ   |
         |                         |  - tenant_id     |
         |                         |  - No ChromaDB   |
         |                         +--------+---------+
         |                                  |
         |                    (internal)   |   (internal)
         |                                  v
         |                         +------------------+
         |                         |  RAG/Agent       |
         |                         |  Worker          |
         |                         |  - ChromaDB client
         |                         |  - OpenAI        |
         |                         |  - OTEL export   |
         |                         +----+-------+-----+
         |                              |       |
         |                    +---------+       +---------+
         |                    |                 |
         |                    v                 v
         |            +-------------+   +------------------+
         |            | ChromaDB    |   | OTEL Collector   |
         |            | (per-tenant |   | (private only)   |
         |            |  collections)|   | :4317/:4318      |
         |            +-------------+   +--------+---------+
         |                                       |
         |                              (HEC TLS)|
         |                                       v
         |                              +------------------+
         |                              | Splunk (VM-B)   |
         |                              | HEC :8088       |
         |                              | UI :8000        |
         +------------------------------+------------------+

Trust boundaries:
  [Internet] --(HTTPS)--> [LB] --(internal)--> [API] --(internal)--> [Worker]
  [Worker] --(internal)--> [ChromaDB], [OTEL Collector]
  [OTEL Collector] --(TLS)--> [Splunk]
  Agent dashboard: same API host but path + authZ; prefer separate ingress.
```

## 2) Risk register

| Issue | Why it matters | Severity | How to detect (observability) | Fix |
|-------|----------------|----------|--------------------------------|-----|
| No customer auth | Cannot attribute or isolate by tenant/user; session hijack | Critical | Logs show no tenant_id/user_id on /chat | Add JWT or API key; require tenant_id in token |
| No agent auth | Any party can act as agent and read all chats | Critical | Audit log for agent login/accept | OIDC/SSO or API key for agent dashboard; audit every sensitive action |
| Single ChromaDB collection | Tenant A can receive Tenant B’s FAQ context | Critical | Trace attributes show cross-tenant retrieval | Per-tenant collection or strict metadata filter + tests |
| Session ID guessable | Attacker reads or injects into other sessions | High | Spike in session IDs from single IP; failed auth | Signed or server-issued session tokens; bind to tenant |
| PII in spans | Compliance and privacy violation in Splunk | High | Search for llm.prompt in Splunk | Redact/hash in SDK or collector; no full prompt/response in spans |
| HEC token in config | Token theft; fake data injection to Splunk | High | Unusual HEC sources; config scan | Move to Vault/secret manager; rotate token |
| OTLP 0.0.0.0, no auth | Fake or malicious telemetry; DoS on Splunk | High | Unknown service names or spike in volume | Bind to internal IP; add OTLP auth or network allowlist |
| TLS skip verify (HEC) | MITM on collector–Splunk link | Medium | Config review | Use proper certs; set insecure_skip_verify: false |
| No rate limiting | DoS and cost abuse across tenants | Medium | Spike in request rate per IP/tenant | WAF or app-level per-tenant/per-IP limits |
| Monolithic container | Large blast radius; hard to scale | Medium | N/A | Split API, Worker, ChromaDB, Collector |
| No audit trail | Cannot prove who did what for compliance | Medium | No audit index in Splunk | Log auth and sensitive actions to audit index |
| CORS allow_origins * | CSRF and unauthorized embedding | Medium | N/A | Restrict origins; use auth and CSRF tokens |
| Secrets in env only | Leak via env dump or image | Medium | N/A | Inject from Vault/secret manager; no secrets in image |

## 3) Proposed architecture steps

**Phase 0 – Quick wins (before multi-tenant)**

- Remove HEC token from YAML/README; use env or secret manager; rotate token.
- Set Splunk HEC TLS `insecure_skip_verify: false`; install correct certs.
- Bind OTEL Collector to internal IP (e.g. 10.0.0.126); restrict firewall to app/worker subnet only.
- Add redaction for span attributes: do not export full `llm.prompt`/`llm.response`; export length/hash or redacted snippet only.
- Add rate limiting on `/chat` (per IP or per session) and optional on agent endpoints.
- Add structured logging with trace_id; no PII in logs.

**Phase 1 – MVP multi-tenant**

- Introduce tenant_id: from JWT or API key (customer) and from SSO/API key (agent). Enforce on every request.
- Session store: key sessions by (tenant_id, session_id); reject requests that don’t match token tenant.
- ChromaDB: move to dedicated service; create per-tenant collection (e.g. `faqs_<tenant_id>`); RAG layer resolves tenant and uses only that collection.
- Propagate tenant_id in OTel resource attributes and baggage; add to logs; redact PII in spans/logs.
- Splunk: add tenant_id to HEC events; configure RBAC/index so tenants see only their data (or separate index per tenant).
- Agent dashboard: require auth; audit log for login, accept, end session.
- Run collector and ChromaDB on internal network only; TLS for HEC.

**Phase 2 – Hardening**

- Split containers: API (customer + optional agent routes), RAG Worker, ChromaDB, OTEL Collector.
- Secrets: full migration to Vault or cloud secret manager; rotation runbook.
- mTLS between Collector and Splunk (or API ↔ Worker) where supported.
- WAF in front of LB; bot protection on chat.
- Per-tenant SLOs and dashboards; sampling and retention policy by tenant tier.
- Optional: per-tenant deployment (Option 3) for high-isolation tenants.

## 4) Action checklist by owner

| # | Action | Owner |
|---|--------|--------|
| 1 | Remove HEC token from repo; move to secret manager; rotate | Security + Infra |
| 2 | Fix TLS verify for Splunk HEC; bind Collector to internal IP | Infra |
| 3 | Add tenant_id to auth (JWT/API key) and enforce on all endpoints | App |
| 4 | Scope session store and queue by tenant_id | App |
| 5 | Move ChromaDB to dedicated service; implement per-tenant collections | App + Infra |
| 6 | Add tenant_id to OTel resource/baggage and logs; redact PII in spans | Observability + App |
| 7 | Splunk: tenant_id in events; RBAC or per-tenant index | Observability |
| 8 | Add rate limiting (per tenant / per IP) and audit logging | App + Security |
| 9 | Authenticate agent dashboard; log agent actions to audit index | App + Security |
| 10 | Split API / Worker / ChromaDB / Collector; update networking | Infra |
| 11 | Document and test secret rotation (HEC, OpenAI, DB) | Security + Infra |
| 12 | Define SLOs and per-tenant dashboards; alerting | Observability |

## 5) Top 10 mistakes to avoid for multi-tenant LLM customer service

1. **Trusting client-supplied tenant_id or session_id** – Always derive tenant (and user) from verified auth (JWT/API key); never from query/body/header without verification.
2. **Single vector collection for all tenants** – Use per-tenant collections or strict metadata + single data access layer; never query without tenant filter.
3. **Logging or tracing full prompts/responses** – Redact or hash PII and sensitive content before sending to any log or tracing backend.
4. **Exposing OTLP publicly without auth** – Bind to internal network and restrict by security group; add auth if OTLP must be reachable from broader network.
5. **Putting secrets in config or README** – Use a secret manager and inject at runtime; rotate after any leak.
6. **Skipping rate limits per tenant** – One abusive tenant can exhaust quota and impact others; enforce per-tenant (and per-user) limits.
7. **No audit trail for agent and admin actions** – Log who did what (login, accept, end session) to an immutable audit store for compliance and forensics.
8. **Mixing customer and agent traffic without stronger auth for agents** – Agent dashboard must require strong auth and be network-restricted; do not rely on “internal only” without auth.
9. **Single monolithic process for UI, API, RAG, and DB** – Split at least API, RAG worker, and ChromaDB to limit blast radius and allow independent scaling.
10. **Forgetting tenant in observability** – Every span and log should carry tenant_id (and be redacted); otherwise debugging and SLOs cannot be tenant-scoped and you risk cross-tenant visibility.

---

## Assumptions made

- OCI is the cloud provider (from SECURITY_Research.md and README); same design applies to AWS/Azure with equivalent VPC/security groups and secret managers.
- “Tenant” = customer organization (retailer); each retailer has its own FAQ corpus and optional agent pool.
- Current deployment is single-node (one app VM/container, one Splunk VM); scaling is future work.
- No existing IdP; JWT or API key per tenant is assumed for Phase 1; OIDC can replace later.
- ChromaDB is acceptable as vector store; migration to managed vector DB is optional and does not change isolation model (per-tenant collection or metadata filter still required).
