# backend/agent.py
# Changes:
# - Load SKILL.md from skills/cannabis-medical-safety/SKILL.md
# - Create a separate "MedicalSafetyAgent" workflow that only runs when a user asks
#   medical/health/dosage/interaction questions (so the skill tokens are only used when needed)
# - Keep the retail/FAQ agent focused on compliant retail/FAQ guidance

import asyncio
import os
import time
import re

# Disable tokenizer parallelism to avoid warnings and potential conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List
from dotenv import load_dotenv
load_dotenv()

# Configuration - REPLACE THESE VALUES
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OTEL: use env so Docker can point to host (e.g. host.docker.internal:4328) or Splunk VM (10.0.0.249:4328)
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://localhost:4328").rstrip("/")
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "openai-sidecar-test")
ENVIRONMENT = os.getenv("OTEL_ENVIRONMENT", "sidecar-agent")

import openai

# Observability Imports with Graceful Degradation
try:
    import openlit
except ImportError:
    openlit = None

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

tracer = None
if OTEL_AVAILABLE:
    try:
        tracer = trace.get_tracer(SERVICE_NAME)
    except Exception:
        tracer = None

# === Initialize OpenLIT ===
all_evals = None
all_guards = None
if openlit:
    try:
        openlit.init(
            otlp_endpoint=OTEL_ENDPOINT,
            disable_metrics=False,
            environment=ENVIRONMENT,
        )
        all_evals = openlit.evals.All(collect_metrics=True)
        all_guards = openlit.guard.All(
            provider="openai",
            api_key=OPENAI_API_KEY,
            collect_metrics=True
        )
        print("OpenLIT initialized successfully")
    except Exception as e:
        print(f"OpenLIT init failed: {e}")

# === Monkeypatch Global OpenAI SDK ===
original_create = openai.chat.completions.create

def patched_create(*args, **kwargs):
    try:
        messages = kwargs.get("messages", [])
        prompt = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else "<no_user_prompt>"

        response = original_create(*args, **kwargs)
        text = response.choices[0].message.content if response.choices else "<empty_response>"

        # === Evaluation ===
        results = None
        if all_evals:
            try:
                results = all_evals.measure(prompt=prompt, text=text, contexts=[])
                print("test2: evaluation done")
            except Exception as e:
                print(f"Evaluation failed: {e}")

        # Use current global TracerProvider (set at startup by setup_splunk_otel).
        # Module-level tracer was created at import time with no-op provider, so spans
        # would never export; get tracer at request time so Splunk receives llm.prompt/llm.response.
        if OTEL_AVAILABLE:
            try:
                current_tracer = trace.get_tracer(SERVICE_NAME)
                with current_tracer.start_as_current_span("openai_sidecar_intercept") as span:
                    span.set_attribute("service.name", SERVICE_NAME)
                    span.set_attribute("llm.prompt", prompt)
                    span.set_attribute("llm.response", text)

                    if results and hasattr(results, 'verdict') and results.verdict != "no":
                        span.set_attribute("eval.verdict", results.verdict)
                        span.set_attribute("eval.evaluation", results.evaluation or "")
                        span.set_attribute("eval.score", results.score or 0)
                        span.set_attribute("eval.classification", results.classification or "")
                        span.set_attribute("eval.explanation", results.explanation or "")
                        print("Evaluation Results:", results)

                    elif all_guards:
                        print("Running guardrail checks...")
                        try:
                            guard_results = all_guards.detect(text=prompt)
                            print("Guardrail Results:", guard_results or "None")

                            if guard_results and guard_results.verdict != "none":
                                span.set_attribute("guard.verdict", guard_results.verdict)
                                span.set_attribute("guard.score", guard_results.score or 0)
                                span.set_attribute("guard.guard", guard_results.guard or "")
                                span.set_attribute("guard.classification", guard_results.classification or "")
                                span.set_attribute("guard.explanation", guard_results.explanation or "")
                        except Exception as e:
                            print(f"Guardrail check failed: {e}")

                    trace.get_tracer_provider().force_flush(timeout_millis=2000)
                    print("test4: span flushed")
            except Exception as e:
                print(f"Tracing failed: {e}")
        return response

    except Exception as e:
        print(f"Sidecar Error: {e}")
        return original_create(*args, **kwargs)

openai.chat.completions.create = patched_create
print("OpenAI SDK patched successfully")
print("Sidecar is now listening...")

from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.tool import Tool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput
from beeai_framework.emitter.emitter import Emitter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb

# Configuration constants for the RAG system
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHROMA_PERSIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "my_chroma_db")
CHROMA_COLLECTION_NAME = "company_faqs"

# === Skill path ===
# backend/agent.py is in backend/, so skills/ is sibling of backend/
SKILLS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
CANNABIS_MEDICAL_SKILL_PATH = os.path.join(SKILLS_DIR, "cannabis-medical-safety", "SKILL.md")

# Global variables
_embedding_model = None
_chroma_collection = None
_llm = None
_faq_tool_instance = None

_agent_workflow = None                 # retail/FAQ workflow
_medical_agent_workflow = None         # medical safety workflow (skill-based)

_tracer_provider = None
_tracer = None

# Cached skill instructions (loaded once)
_medical_skill_instructions = None


def _load_skill_instructions(skill_path: str) -> str:
    """Load a SKILL.md file (manifest/instructions) from disk."""
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fail safe: if skill missing, return a minimal safe policy.
        # (So production doesn't crash if the file isn't mounted in Docker.)
        return (
            "---\n"
            "name: cannabis-medical-safety\n"
            "description: Safety fallback skill; file missing.\n"
            "---\n\n"
            "# Cannabis Medical Information Skill (Fallback)\n"
            "If a user asks for medical advice, respond with a disclaimer and advise consulting a licensed clinician.\n"
            "Do not provide dosage numbers or medical claims.\n"
        )
    except Exception as e:
        return (
            "---\n"
            "name: cannabis-medical-safety\n"
            "description: Safety fallback skill; file unreadable.\n"
            "---\n\n"
            f"# Error loading skill: {type(e).__name__}\n"
            "Respond with a disclaimer and recommend consulting a licensed clinician.\n"
            "Do not provide dosage numbers or medical claims.\n"
        )

def _normalize(text: str) -> str:
    """Make matching easier (hyphens/spaces/etc.)."""
    t = (text or "").lower()
    t = t.replace("-", " ")               # self-harm -> self harm
    t = re.sub(r"\s+", " ", t).strip()    # collapse extra spaces
    return t

ESCALATION_TERMS = [
    # Pregnancy / breastfeeding
    "pregnant", "pregnancy", "expecting", "trimester", "preggo",
    "breastfeeding", "breast feeding", "nursing", "lactating",

    # Chest pain / heart danger
    "chest pain", "heart attack", "tightness in chest", "pressure in chest",
    "palpitations", "heart racing",

    # Seizures
    "seizure", "seizures", "convulsion", "convulsions",

    # Self-harm / suicide
    "suicidal", "self harm", "harm myself", "kill myself", "want to die", "end my life",

    # Pediatric
    "pediatric", "child", "kid", "teen", "minor", "under 18", "my son", "my daughter",
]

def _is_medical_skill_query(user_query: str) -> bool:
    """
    Router: if query requires clinical guardrails (dosing, drug interactions,
    escalation emergencies), route to medical safety skill workflow.
    
    General wellness/lifestyle queries (sleep, anxiety, relaxation) flow straight
    to LLM to receive non-medical descriptive responses per sponsor guidelines.
    """
    q = _normalize(user_query)

    # Gate 1 — hard escalation terms (defined externally, non-negotiable)
    if any(term in q for term in ESCALATION_TERMS):
        return True

    # Gate 2 — strictly clinical/dosing terms only
    hard_clinical = [
        "dose", "dosage", "mg", "milligram", "titrate",
        "interaction", "contraindication", "cyp", "cyp450",
        "overdose", "withdrawal", "seizure", "epilepsy",
        "blood pressure", "hypertension", "ssri", "warfarin",
        "blood thinner", "pharmacist", "prescription",
    ]
    if any(t in q for t in hard_clinical):
        return True

    # Gate 3 — "strain for [health topic]" compound rule (narrow, intentional)
    health_topics = [
        "sleep", "insomnia", "anxiety", "panic", "pain",
        "inflammation", "nausea", "ptsd", "depression",
        "arthritis", "cancer", "migraine", "adhd",
    ]
    if "strain" in q and any(t in q for t in health_topics):
        return True

    # Gate 4 — clinical sleep queries only (sponsor flagged "sleep" as missing)
    # casual sleep queries ("help me sleep", "good for sleep") are intentionally
    # excluded and will receive non-medical lifestyle responses
    sleep_clinical = [
        "sleep aid", "sleeping pill", "melatonin dose", "how much melatonin"
    ]
    if any(t in q for t in sleep_clinical):
        return True

    return False


def setup_splunk_otel():
    if not OTEL_AVAILABLE:
        print("OpenTelemetry not available - skipping OTEL setup")
        return None

    try:
        base = os.getenv("OTEL_ENDPOINT", "http://localhost:4328").rstrip("/")
        otel_traces_url = base if base.endswith("/v1/traces") else f"{base}/v1/traces"
        svc_name = os.getenv("OTEL_SERVICE_NAME", "beeai-faq-agent")
        env_name = os.getenv("OTEL_ENVIRONMENT", "production")

        print(f"Setting up Splunk SignalFX OTEL integration...")
        print(f"   Endpoint: {otel_traces_url}")
        print(f"   Service: {svc_name}")
        print(f"   Environment: {env_name}")

        resource = Resource.create({
            "service.name": svc_name,
            "service.version": "1.0.0",
            "deployment.environment": env_name,
            "telemetry.sdk.name": "beeai-framework",
            "telemetry.sdk.version": "0.1.17"
        })

        tracer_provider = TracerProvider(resource=resource)

        otlp_exporter = OTLPSpanExporter(
            endpoint=otel_traces_url,
            headers={}
        )

        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(tracer_provider)

        print("Splunk SignalFX OTEL integration configured successfully")
        return tracer_provider

    except Exception as e:
        print(f"Failed to configure Splunk SignalFX OTEL: {e}")
        print("Check your OTEL endpoint and configuration")
        return None


def test_span_export(tracer_provider, endpoint: str):
    try:
        print(f"Testing span export to {endpoint}...")

        test_tracer = trace.get_tracer("span-test")
        with test_tracer.start_as_current_span("test_span") as span:
            span.set_attribute("test.attribute", "span_export_test")
            span.set_attribute("test.timestamp", time.time())
            span.set_attribute("test.service", "beeai-faq-agent")

        tracer_provider.force_flush(timeout_millis=3000)

        print("Span export test completed successfully")
        print("Check your Splunk dashboard for the test span")
        return True

    except Exception as e:
        print(f"Span export test failed: {e}")
        print("Check your OTEL endpoint connectivity")
        return False


def _setup_rag_system():
    global _embedding_model, _chroma_collection, _llm, _faq_tool_instance
    global _agent_workflow, _medical_agent_workflow
    global _tracer_provider, _tracer
    global _medical_skill_instructions

    # Note: we now have TWO workflows; both must exist to say "already setup".
    if (_embedding_model and _chroma_collection and _llm and _faq_tool_instance and
        _agent_workflow and _medical_agent_workflow and _medical_skill_instructions):
        print("RAG system already set up.")
        return True

    print("Setting up RAG system components...")

    if OTEL_AVAILABLE:
        try:
            _tracer_provider = setup_splunk_otel()
            if _tracer_provider:
                _tracer = trace.get_tracer("beeai-faq-agent")
                global tracer
                tracer = trace.get_tracer(SERVICE_NAME)
                print("OpenTelemetry tracer initialized for observability")

                if os.getenv("OTEL_SKIP_SPAN_TEST", "1") != "0":
                    print("Skipping span export test (set OTEL_SKIP_SPAN_TEST=0 to enable)")
                else:
                    otel_endpoint = os.getenv("OTEL_ENDPOINT", "http://localhost:4328")
                    span_test_success = test_span_export(_tracer_provider, otel_endpoint)
                    if not span_test_success:
                        print("Warning: Span export test failed - traces may not be reaching Splunk")
                    else:
                        print("Span export test passed - traces are being sent to Splunk successfully")
            else:
                print("OpenTelemetry setup failed - continuing without observability")
        except Exception as e:
            print(f"Error setting up OpenTelemetry: {e}")
            print("Continuing without observability")
    else:
        print("OpenTelemetry not available - running without observability")

    # Load skill text once (keeps it out of base prompt unless used)
    _medical_skill_instructions = _load_skill_instructions(CANNABIS_MEDICAL_SKILL_PATH)
    if os.path.exists(CANNABIS_MEDICAL_SKILL_PATH):
        print(f"Loaded medical skill from {CANNABIS_MEDICAL_SKILL_PATH}")
    else:
        print(f"Medical skill file not found at {CANNABIS_MEDICAL_SKILL_PATH} - using fallback skill text")

    try:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return False

    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        _chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready with {_chroma_collection.count()} documents.")
        print("CHROMA_PERSIST_PATH =", CHROMA_PERSIST_PATH)
        print("CHROMA_COLLECTION_NAME =", CHROMA_COLLECTION_NAME)
        print("CHROMA_DOC_COUNT =", _chroma_collection.count())
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return False

    try:
        _llm = ChatModel.from_name(os.environ.get("OPENAI_MODEL", "openai:gpt-4o"))
        print("OpenAI LLM initialized.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return False

    _faq_tool_instance = FAQTool(embedding_model=_embedding_model, chroma_collection=_chroma_collection)
    print("FAQTool instance created.")

    # --- Retail/FAQ workflow (kept lean; no long “ban list” needed if we route medical queries away) ---
    _agent_workflow = AgentWorkflow(name="Company FAQ Assistant")
    _agent_workflow.add_agent(
        name="FAQAgent",
        role="A fun, friendly Washington cannabis budtender focused on compliance-safe FAQ guidance.",
        instructions=(
            "You are a Washington State cannabis retail compliance assistant with a fun budtender vibe. "
            "Your primary goal is to answer using only the provided FAQ context. "
            "If no relevant FAQ context is provided, say you cannot find it in the store FAQs. "

            "Do NOT provide medical advice, dosing advice, or therapeutic health claims. "
            "Never state that cannabis treats, cures, prevents, or alleviates medical conditions. "

            "Allowed content includes flavor/aroma descriptions, factual cannabinoid and terpene information, "
            "potency details, consumption methods, and neutral consumer-reported experiences "
            "(example: customers often describe this as uplifting or relaxing). "

            "If a user asks a health or medical question, those questions are handled by a separate "
            "medical safety skill, so do not answer them here."

            "Do NOT try to use the 'faq_lookup_tool' on your own if context is already provided."
        ),
        tools=[_faq_tool_instance],
        llm=_llm,
    )

    # --- Medical safety workflow (skill-based; only invoked on medical-ish queries) ---
    _medical_agent_workflow = AgentWorkflow(name="Cannabis Medical Safety")
    _medical_agent_workflow.add_agent(
        name="MedicalSafetyAgent",
        role="A cannabis medical information safety specialist.",
        instructions=_medical_skill_instructions,
        tools=[],   # intentionally no tools here; your SKILL.md already says no numeric dosing unless retrieved.
        llm=_llm,
    )

    print("Agent workflows created: FAQAgent + MedicalSafetyAgent (skill-based).")
    return True


class FAQTool(Tool):
    name: str = "faq_lookup_tool"
    description: str = (
        "Searches the company's frequently asked questions for relevant answers using semantic search. "
        "Use this tool when the user asks a question about company policies, products, or general FAQs. "
        "Input should be a question string."
    )

    class FAQToolInput(BaseModel):
        query: str = Field(description="The question to lookup in the company FAQs.")

    @property
    def input_schema(self) -> type[BaseModel]:
        return self.FAQToolInput

    def _create_emitter(self) -> Emitter | None:
        return Emitter()

    def __init__(self, embedding_model: SentenceTransformer, chroma_collection: chromadb.Collection):
        super().__init__()
        self.embedding_model = embedding_model
        self.chroma_collection = chroma_collection

    async def _run(self, query: str) -> str:
        if _tracer:
            with _tracer.start_as_current_span("faq_tool_execution") as span:
                span.set_attribute("tool.name", self.name)
                span.set_attribute("tool.description", self.description)
                span.set_attribute("faq.query", query)
                span.set_attribute("faq.query_length", len(query))

                try:
                    with _tracer.start_as_current_span("query_embedding") as embedding_span:
                        embedding_span.set_attribute("embedding.model", "sentence-transformers")
                        embedding_span.set_attribute("embedding.model_name", "all-MiniLM-L6-v2")

                        query_embedding = self.embedding_model.encode(query).tolist()
                        embedding_span.set_attribute("embedding.vector_size", len(query_embedding))
                        embedding_span.set_attribute("embedding.success", True)

                    with _tracer.start_as_current_span("chroma_search") as search_span:
                        search_span.set_attribute("chroma.collection", CHROMA_COLLECTION_NAME)
                        search_span.set_attribute("chroma.n_results", 3)

                        results = self.chroma_collection.query(
                            query_embeddings=[query_embedding],
                            n_results=3,
                            include=['documents', 'metadatas']
                        )

                        search_span.set_attribute(
                            "chroma.results_count",
                            len(results.get('documents', [[]])[0]) if results and results.get('documents') else 0
                        )
                        search_span.set_attribute("chroma.search_success", True)

                    with _tracer.start_as_current_span("result_processing") as process_span:
                        retrieved_contexts = []
                        if results and results.get('documents') and results['documents'][0]:
                            for i in range(len(results['documents'][0])):
                                doc_content = results['documents'][0][i]
                                metadata = results['metadatas'][0][i]
                                question = metadata.get('question', 'N/A')
                                answer = metadata.get('answer', doc_content)
                                retrieved_contexts.append(f"Question: {question}\nAnswer: {answer}")

                        process_span.set_attribute("processing.contexts_count", len(retrieved_contexts))
                        process_span.set_attribute("processing.success", True)

                    if not retrieved_contexts:
                        span.set_attribute("faq.no_results", True)
                        span.set_attribute("faq.status", "no_results")
                        return "No relevant information found in the FAQs."

                    context_string = "\n\n".join(retrieved_contexts)
                    span.set_attribute("faq.results_found", True)
                    span.set_attribute("faq.status", "success")
                    span.set_attribute("faq.response_length", len(context_string))

                    return context_string

                except Exception as e:
                    span.set_attribute("faq.status", "error")
                    span.set_attribute("error.message", str(e))
                    span.set_attribute("error.type", type(e).__name__)

                    print(f"Error in FAQ tool execution: {e}")
                    return f"Error processing query for FAQ lookup: {e}"
        else:
            try:
                query_embedding = self.embedding_model.encode(query).tolist()

                results = self.chroma_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=3,
                    include=['documents', 'metadatas']
                )

                retrieved_contexts = []
                if results and results.get('documents') and results['documents'][0]:
                    for i in range(len(results['documents'][0])):
                        doc_content = results['documents'][0][i]
                        metadata = results['metadatas'][0][i]
                        question = metadata.get('question', 'N/A')
                        answer = metadata.get('answer', doc_content)
                        retrieved_contexts.append(f"Question: {question}\nAnswer: {answer}")

                if not retrieved_contexts:
                    return "No relevant information found in the FAQs."

                context_string = "\n\n".join(retrieved_contexts)
                return context_string

            except Exception as e:
                return f"Error processing query for FAQ lookup: {e}"


async def run_faq_agent(user_query: str) -> str:
    """
    Main entrypoint.
    - If query is medical/health/dosage/interaction related, route to MedicalSafetyAgent (skill-based).
    - Otherwise run retail FAQ RAG + FAQAgent.
    """
    if _tracer:
        with _tracer.start_as_current_span("faq_agent_workflow") as span:
            span.set_attribute("workflow.name", "Company FAQ Assistant")
            span.set_attribute("workflow.query", user_query)
            span.set_attribute("workflow.query_length", len(user_query))
            span.set_attribute("workflow.timestamp", time.time())

            try:
                with _tracer.start_as_current_span("rag_system_setup") as setup_span:
                    if not _setup_rag_system():
                        setup_span.set_attribute("setup.status", "failed")
                        span.set_attribute("workflow.status", "setup_failed")
                        return "Backend RAG system failed to initialize. Please check server logs."
                    setup_span.set_attribute("setup.status", "success")

                # === Skill routing ===
                use_medical_skill = _is_medical_skill_query(user_query)
                span.set_attribute("routing.medical_skill", use_medical_skill)

                if use_medical_skill:
                    # No FAQ retrieval; directly answer using skill constraints
                    with _tracer.start_as_current_span("medical_skill_execution") as med_span:
                        med_span.set_attribute("agent.name", "MedicalSafetyAgent")

                        response = await _medical_agent_workflow.run(
                            inputs=[AgentWorkflowInput(prompt=user_query)]
                        )

                        final_answer = response.result.final_answer
                        med_span.set_attribute("workflow.response_length", len(final_answer))
                        med_span.set_attribute("workflow.success", True)

                        # Useful for SIEM searches
                        max_len = int(os.getenv("OTEL_TEXT_MAX_LEN", "4096"))
                        if max_len > 0:
                            prompt_attr = user_query if len(user_query) <= max_len else (user_query[:max_len] + "…")
                            answer_attr = final_answer if len(final_answer) <= max_len else (final_answer[:max_len] + "…")
                            span.set_attribute("llm.prompt", prompt_attr)
                            span.set_attribute("llm.response", answer_attr)
                            span.set_attribute("llm.response_length", len(final_answer))

                        span.set_attribute("workflow.status", "success")
                        return final_answer

                # === Retail FAQ path ===
                print("Calling the faq_lookup_tool...")
                with _tracer.start_as_current_span("faq_tool_execution") as tool_span:
                    tool_span.set_attribute("tool.name", "faq_lookup_tool")
                    retrieved_info = await _faq_tool_instance._run(user_query)
                    tool_span.set_attribute("tool.response_length", len(retrieved_info))
                    tool_span.set_attribute("tool.success", True)

                prompt_for_agent = f"Retrieved Company FAQ Information:\n{retrieved_info}\n\nUser Question: {user_query}"

                with _tracer.start_as_current_span("prompt_construction") as prompt_span:
                    prompt_span.set_attribute("prompt.retrieved_info_length", len(retrieved_info))
                    prompt_span.set_attribute("prompt.final_length", len(prompt_for_agent))
                    prompt_span.set_attribute("prompt.construction_success", True)

                with _tracer.start_as_current_span("agent_workflow_execution") as workflow_span:
                    workflow_span.set_attribute("workflow.agent_name", "FAQAgent")
                    workflow_span.set_attribute("workflow.llm_model", os.environ.get("OPENAI_MODEL", "openai:gpt-4o"))

                    response = await _agent_workflow.run(
                        inputs=[AgentWorkflowInput(prompt=prompt_for_agent)]
                    )

                    final_answer = response.result.final_answer
                    workflow_span.set_attribute("workflow.response_length", len(final_answer))
                    workflow_span.set_attribute("workflow.success", True)
                    span.set_attribute("workflow.status", "success")

                    max_len = int(os.getenv("OTEL_TEXT_MAX_LEN", "4096"))
                    if max_len > 0:
                        prompt_attr = user_query if len(user_query) <= max_len else (user_query[:max_len] + "…")
                        answer_attr = final_answer if len(final_answer) <= max_len else (final_answer[:max_len] + "…")
                        span.set_attribute("llm.prompt", prompt_attr)
                        span.set_attribute("llm.response", answer_attr)
                        span.set_attribute("llm.response_length", len(final_answer))

                    return final_answer

            except Exception as e:
                span.set_attribute("workflow.status", "error")
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)

                print(f"Error running agent workflow: {e}")
                return f"An error occurred while processing your request: {e}"

    # === No tracer path ===
    if not _setup_rag_system():
        return "Backend RAG system failed to initialize. Please check server logs."

    if _is_medical_skill_query(user_query):
        try:
            response = await _medical_agent_workflow.run(
                inputs=[AgentWorkflowInput(prompt=user_query)]
            )
            return response.result.final_answer
        except Exception as e:
            print(f"Error running medical skill workflow: {e}")
            return f"An error occurred while processing your request: {e}"

    print("Calling the faq_lookup_tool...")
    retrieved_info = await _faq_tool_instance._run(user_query)
    prompt_for_agent = f"Retrieved Company FAQ Information:\n{retrieved_info}\n\nUser Question: {user_query}"

    try:
        response = await _agent_workflow.run(
            inputs=[AgentWorkflowInput(prompt=prompt_for_agent)]
        )
        return response.result.final_answer
    except Exception as e:
        print(f"Error running agent workflow: {e}")
        return f"An error occurred while processing your request: {e}"


def get_observability_data() -> dict:
    return {
        "rag_system": {
            "embedding_model_loaded": _embedding_model is not None,
            "chroma_collection_ready": _chroma_collection is not None,
            "llm_initialized": _llm is not None,
            "faq_tool_ready": _faq_tool_instance is not None,
            "agent_workflow_ready": _agent_workflow is not None,
            "medical_skill_workflow_ready": _medical_agent_workflow is not None,
            "medical_skill_loaded": _medical_skill_instructions is not None,
            "medical_skill_path": CANNABIS_MEDICAL_SKILL_PATH,
        },
        "opentelemetry": {
            "available": OTEL_AVAILABLE,
            "tracer_provider_ready": _tracer_provider is not None,
            "tracer_ready": _tracer is not None
        },
        "chroma_db": {
            "collection_name": CHROMA_COLLECTION_NAME,
            "document_count": _chroma_collection.count() if _chroma_collection else 0,
            "persist_path": CHROMA_PERSIST_PATH
        },
        "embedding_model": {
            "name": EMBEDDING_MODEL_NAME,
            "loaded": _embedding_model is not None
        },
        "status": "ready" if all([
            _embedding_model, _chroma_collection, _llm,
            _faq_tool_instance, _agent_workflow, _medical_agent_workflow,
            _medical_skill_instructions
        ]) else "initializing"
    }