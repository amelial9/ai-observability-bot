# backend/agent.py
# This file implements a RAG (Retrieval-Augmented Generation) system for company FAQs
# using the BeeAI framework, ChromaDB for vector storage, and OpenAI for LLM responses.

import asyncio
import os
import time
# Disable tokenizer parallelism to avoid warnings and potential conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List
from dotenv import load_dotenv
load_dotenv()

# Configuration - REPLACE THESE VALUES
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OTEL_ENDPOINT = "http://localhost:4328"
SERVICE_NAME = "openai-sidecar-test"
ENVIRONMENT = "sidecar-agent"

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

        if OTEL_AVAILABLE and tracer:
             try:
                 with tracer.start_as_current_span("openai_sidecar_intercept") as span:
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
# Use absolute path for Docker compatibility
CHROMA_PERSIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "my_chroma_db")
CHROMA_COLLECTION_NAME = "company_faqs"

# Global variables
_embedding_model = None
_chroma_collection = None
_llm = None
_faq_tool_instance = None
_agent_workflow = None
_tracer_provider = None
_tracer = None

def setup_splunk_otel():
    if not OTEL_AVAILABLE:
        print("OpenTelemetry not available - skipping OTEL setup")
        return None
    
    try:
        OTEL_ENDPOINT = "http://localhost:4328/v1/traces"
        SERVICE_NAME = "beeai-faq-agent"
        ENVIRONMENT = "production"
        
        print(f"Setting up Splunk SignalFX OTEL integration...")
        print(f"   Endpoint: {OTEL_ENDPOINT}")
        print(f"   Service: {SERVICE_NAME}")
        print(f"   Environment: {ENVIRONMENT}")
        
        resource = Resource.create({
            "service.name": SERVICE_NAME,
            "service.version": "1.0.0",
            "deployment.environment": ENVIRONMENT,
            "telemetry.sdk.name": "beeai-framework",
            "telemetry.sdk.version": "0.1.17"
        })
        
        tracer_provider = TracerProvider(resource=resource)
        
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTEL_ENDPOINT,
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
        
        # Use short timeout to avoid blocking 30s when OTEL collector is unreachable (e.g. in Docker)
        tracer_provider.force_flush(timeout_millis=3000)
        
        print("Span export test completed successfully")
        print("Check your Splunk dashboard for the test span")
        return True
        
    except Exception as e:
        print(f"Span export test failed: {e}")
        print("Check your OTEL endpoint connectivity")
        return False

def _setup_rag_system():
    global _embedding_model, _chroma_collection, _llm, _faq_tool_instance, _agent_workflow, _tracer_provider, _tracer

    if _embedding_model and _chroma_collection and _llm and _faq_tool_instance and _agent_workflow:
        print("RAG system already set up.")
        return True

    print("Setting up RAG system components...")
    
    if OTEL_AVAILABLE:
        try:
            _tracer_provider = setup_splunk_otel()
            if _tracer_provider:
                _tracer = trace.get_tracer("beeai-faq-agent")
                print("OpenTelemetry tracer initialized for observability")
                
                # Skip span export test during setup - it can block 30s if OTEL collector unreachable (e.g. Docker).
                # Traces will still export; test is optional. Set OTEL_SKIP_SPAN_TEST=0 to run it.
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

    _agent_workflow = AgentWorkflow(name="Company FAQ Assistant")
    _agent_workflow.add_agent(
        name="FAQAgent",
        role="An expert in company FAQs.",
        instructions=(
            "You are an expert in company FAQs. Your primary goal is to answer questions based on the provided company FAQ information. "
            "If company FAQ information is provided in the input, prioritize using it to answer the user's question. "
            "If no relevant company FAQ information is provided or found, state that you cannot find the answer in the company FAQs. "
            "Do NOT try to use the 'faq_lookup_tool' on your own if context is already provided, as the information has already been retrieved for you."
        ),
        tools=[_faq_tool_instance],
        llm=_llm,
    )
    print("Agent workflow created and agent added.")
    return True

class FAQTool(Tool):
    name: str = "faq_lookup_tool"
    description: str = "Searches the company's frequently asked questions for relevant answers using semantic search. Use this tool when the user asks a question about company policies, products, or general FAQs. Input should be a question string."

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
                        
                        search_span.set_attribute("chroma.results_count", len(results.get('documents', [[]])[0]) if results and results.get('documents') else 0)
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
                    workflow_span.set_attribute("workflow.llm_model", "openai:gpt-4o")
                    
                    response = await _agent_workflow.run(
                        inputs=[
                            AgentWorkflowInput(
                                prompt=prompt_for_agent,
                            )
                        ]
                    )
                    
                    final_answer = response.result.final_answer
                    workflow_span.set_attribute("workflow.response_length", len(final_answer))
                    workflow_span.set_attribute("workflow.success", True)
                    span.set_attribute("workflow.status", "success")
                    
                    return final_answer
                    
            except Exception as e:
                span.set_attribute("workflow.status", "error")
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                
                print(f"Error running agent workflow: {e}")
                return f"An error occurred while processing your request: {e}"
    else:
        if not _setup_rag_system():
            return "Backend RAG system failed to initialize. Please check server logs."

        print("Calling the faq_lookup_tool...")
        retrieved_info = await _faq_tool_instance._run(user_query)
       
        prompt_for_agent = f"Retrieved Company FAQ Information:\n{retrieved_info}\n\nUser Question: {user_query}"

        try:
            response = await _agent_workflow.run(
                inputs=[
                    AgentWorkflowInput(
                        prompt=prompt_for_agent,
                    )
                ]
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
            "agent_workflow_ready": _agent_workflow is not None
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
            _faq_tool_instance, _agent_workflow
        ]) else "initializing"
    }