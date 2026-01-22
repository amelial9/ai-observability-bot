# backend/agent.py
# This file implements a RAG (Retrieval-Augmented Generation) system for company FAQs
# using the BeeAI framework, ChromaDB for vector storage, and OpenAI for LLM responses.

import asyncio
import os
import time
# Disable tokenizer parallelism to avoid warnings and potential conflicts
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List

import openai
import openlit
# from openai import OpenAI
from opentelemetry import trace

from dotenv import load_dotenv
load_dotenv()

# Configuration - REPLACE THESE VALUES
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Your test OpenAI key
OTEL_ENDPOINT = "http://localhost:4328"  # OTEL endpoint in Docker
SERVICE_NAME = "openai-sidecar-test"  # Identify in Splunk
ENVIRONMENT = "sidecar-agent"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# === Initialize OpenLIT ===
try:
    openlit.init(
        otlp_endpoint=OTEL_ENDPOINT,
        disable_metrics=False,  # Metric collection is enabled
        environment=ENVIRONMENT,
    )
    all_evals = openlit.evals.All(collect_metrics=True)
    all_guards = openlit.guard.All(
        provider="openai",
        api_key=OPENAI_API_KEY,
        collect_metrics=True
    )
    print("✅ OpenLIT initialized successfully")
except Exception as e:
    print(f"❌ OpenLIT init failed: {e}")
    exit(1)

# === OpenTelemetry Tracer ===
tracer = trace.get_tracer(SERVICE_NAME)

# === Monkeypatch OpenAI SDK ===
# client = OpenAI(api_key=OPENAI_API_KEY)
# original_create = client.chat.completions.create
# === Monkeypatch Global OpenAI SDK ===
original_create = openai.chat.completions.create

def patched_create(*args, **kwargs):
    try:
        messages = kwargs.get("messages", [])
        prompt = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else "<no_user_prompt>"

        response = original_create(*args, **kwargs)
        text = response.choices[0].message.content if response.choices else "<empty_response>"
        # text= "The current president of the moon is Neil Armstrong."
        # === Evaluation ===
        results = all_evals.measure(prompt=prompt, text=text, contexts=[])
        print("✅ test2: evaluation done")

        with tracer.start_as_current_span("openai_sidecar_intercept") as span:
            span.set_attribute("service.name", SERVICE_NAME)
            span.set_attribute("llm.prompt", prompt)
            span.set_attribute("llm.response", text)

            if results and results.verdict != "no":
                span.set_attribute("eval.verdict", results.verdict)
                span.set_attribute("eval.evaluation", results.evaluation or "")
                span.set_attribute("eval.score", results.score or 0)
                span.set_attribute("eval.classification", results.classification or "")
                span.set_attribute("eval.explanation", results.explanation or "")
                print("📊 Evaluation Results:", results)

            else:
                print("🔍 Running guardrail checks...")
                guard_results = all_guards.detect(text=prompt)
                print("🛡️ Guardrail Results:", guard_results or "None")

                if guard_results and guard_results.verdict != "none":
                    span.set_attribute("guard.verdict", guard_results.verdict)
                    span.set_attribute("guard.score", guard_results.score or 0)
                    span.set_attribute("guard.guard", guard_results.guard or "")
                    span.set_attribute("guard.classification", guard_results.classification or "")
                    span.set_attribute("guard.explanation", guard_results.explanation or "")

            trace.get_tracer_provider().force_flush()
            print("✅ test4: span flushed")
            return response

    except Exception as e:
        print(f"❌ Sidecar Error: {e}")
        return original_create(*args, **kwargs)

# === Patch SDK ===
openai.chat.completions.create = patched_create
print("🛠️ OpenAI SDK patched successfully")
print("👂 Sidecar is now listening...")


# === Optional Test Trigger ===
if __name__ == "__main__":
    print("🚀 Sidecar daemon is now running and listening for OpenAI calls...")
    try:

        while True:
            time.sleep(60)

    except KeyboardInterrupt:
        print("🛑 Sidecar daemon stopped.")
    except Exception as e:
        print(f"❌ Runtime error: {e}")




# OpenTelemetry imports for observability
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
    print("✅ OpenTelemetry available for observability")
except ImportError:
    OTEL_AVAILABLE = False
    print("⚠️  OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http")

# Import BeeAI framework components for building the agent system
from beeai_framework.backend.chat import ChatModel  # For LLM integration (OpenAI)
from beeai_framework.tools.tool import Tool  # Base class for creating tools
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput  # Workflow management
from beeai_framework.emitter.emitter import Emitter  # Event emission system
from pydantic import BaseModel, Field  # Data validation and schema definition

# Import external libraries for RAG functionality
from sentence_transformers import SentenceTransformer  # For creating text embeddings
import chromadb  # Vector database for storing and searching FAQ embeddings

# Configuration constants for the RAG system
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Lightweight but effective embedding model
# Use path relative to project root (where uvicorn is started from)
CHROMA_PERSIST_PATH = "my_chroma_db"  # Local path for ChromaDB storage
CHROMA_COLLECTION_NAME = "company_faqs"  # Collection name in ChromaDB

# Global variables to store initialized components (singleton pattern)
# These are initialized once and reused across multiple requests
_embedding_model = None  # SentenceTransformer model for creating embeddings
_chroma_collection = None  # ChromaDB collection for FAQ storage
_llm = None  # OpenAI chat model instance
_faq_tool_instance = None  # FAQTool instance for searching FAQs
_agent_workflow = None  # AgentWorkflow for orchestrating the FAQ agent
_tracer_provider = None  # OpenTelemetry tracer provider for observability
_tracer = None  # OpenTelemetry tracer for creating spans

def setup_splunk_otel():
    """Set up OpenTelemetry for Splunk SignalFX observability."""
    if not OTEL_AVAILABLE:
        print("⚠️  OpenTelemetry not available - skipping OTEL setup")
        return None

    try:
        # Configuration from environment variables
        OTEL_ENDPOINT = "http://localhost:4328/v1/traces"
        SERVICE_NAME = "beeai-faq-agent"
        ENVIRONMENT = "production"

        print(f"🔧 Setting up Splunk SignalFX OTEL integration...")
        print(f"   Endpoint: {OTEL_ENDPOINT}")
        print(f"   Service: {SERVICE_NAME}")
        print(f"   Environment: {ENVIRONMENT}")

        # Create resource with service information
        resource = Resource.create({
            "service.name": SERVICE_NAME,
            "service.version": "1.0.0",
            "deployment.environment": ENVIRONMENT,
            "telemetry.sdk.name": "beeai-framework",
            "telemetry.sdk.version": "0.1.17"
        })

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)

        # Create OTLP exporter for Splunk SignalFX
        otlp_exporter = OTLPSpanExporter(
            endpoint=OTEL_ENDPOINT,
            headers={
                # Add any required headers for your Splunk setup
                # "Authorization": "Bearer your-token",
                # "X-SF-TOKEN": "your-signalfx-token"
            }
        )

        # Add batch processor
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(tracer_provider)

        print("✅ Splunk SignalFX OTEL integration configured successfully")
        return tracer_provider

    except Exception as e:
        print(f"⚠️  Failed to configure Splunk SignalFX OTEL: {e}")
        print("💡 Check your OTEL endpoint and configuration")
        return None

def test_span_export(tracer_provider, endpoint: str):
    """Test that spans can be exported successfully to verify connectivity."""
    try:
        print(f"🧪 Testing span export to {endpoint}...")

        # Create a test span
        test_tracer = trace.get_tracer("span-test")
        with test_tracer.start_as_current_span("test_span") as span:
            span.set_attribute("test.attribute", "span_export_test")
            span.set_attribute("test.timestamp", time.time())
            span.set_attribute("test.service", "beeai-faq-agent")

        # Force export by shutting down the tracer provider
        tracer_provider.force_flush()

        print("✅ Span export test completed successfully")
        print("💡 Check your Splunk dashboard for the test span")
        return True

    except Exception as e:
        print(f"❌ Span export test failed: {e}")
        print("💡 Check your OTEL endpoint connectivity")
        return False

def _setup_rag_system():
    """
    Initializes all RAG system components and stores them globally.

    This function implements a singleton pattern - it only initializes components
    once and reuses them for subsequent requests, improving performance.

    Returns:
        bool: True if setup successful, False otherwise
    """
    global _embedding_model, _chroma_collection, _llm, _faq_tool_instance, _agent_workflow, _tracer_provider, _tracer

    # Check if all components are already initialized
    if _embedding_model and _chroma_collection and _llm and _faq_tool_instance and _agent_workflow:
        print("RAG system already set up.")
        return True

    print("Setting up RAG system components...")

    # Step 0: Initialize OpenTelemetry for observability
    if OTEL_AVAILABLE:
        try:
            _tracer_provider = setup_splunk_otel()
            if _tracer_provider:
                _tracer = trace.get_tracer("beeai-faq-agent")
                print("✅ OpenTelemetry tracer initialized for observability")

                # Test span export to verify connectivity
                otel_endpoint = os.getenv("OTEL_ENDPOINT", "http://localhost:4328")
                span_test_success = test_span_export(_tracer_provider, otel_endpoint)
                if not span_test_success:
                    print("⚠️  Warning: Span export test failed - traces may not be reaching Splunk")
                else:
                    print("✅ Span export test passed - traces are being sent to Splunk successfully")
            else:
                print("⚠️  OpenTelemetry setup failed - continuing without observability")
        except Exception as e:
            print(f"⚠️  Error setting up OpenTelemetry: {e}")
            print("💡 Continuing without observability")
    else:
        print("⚠️  OpenTelemetry not available - running without observability")

    # Step 1: Initialize the embedding model for converting text to vectors
    try:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return False

    # Step 2: Initialize ChromaDB for vector storage and retrieval
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        _chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready with {_chroma_collection.count()} documents.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return False

    # Step 3: Initialize the OpenAI LLM for generating responses
    try:
        _llm = ChatModel.from_name(os.environ.get("OPENAI_MODEL", "openai:gpt-4o"))
        print("OpenAI LLM initialized.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return False

    # Step 4: Create the FAQ tool instance that combines embedding and search
    _faq_tool_instance = FAQTool(embedding_model=_embedding_model, chroma_collection=_chroma_collection)
    print("FAQTool instance created.")

    # Step 5: Create the agent workflow that orchestrates the FAQ answering process
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
        tools=[_faq_tool_instance],  # Give the agent access to the FAQ search tool
        llm=_llm,  # Provide the LLM for generating responses
    )
    print("Agent workflow created and agent added.")
    return True

class FAQTool(Tool):
    """
    A custom tool that extends the BeeAI Tool class to provide FAQ search functionality.

    This tool uses semantic search to find relevant FAQ information by:
    1. Converting user queries to embeddings
    2. Searching the ChromaDB collection for similar FAQ embeddings
    3. Returning the most relevant FAQ content
    """

    # Tool metadata for the BeeAI framework
    name: str = "faq_lookup_tool"
    description: str = "Searches the company's frequently asked questions for relevant answers using semantic search. Use this tool when the user asks a question about company policies, products, or general FAQs. Input should be a question string."

    class FAQToolInput(BaseModel):
        """
        Input schema for the FAQ tool, defining the expected input structure.

        This ensures that the tool receives properly formatted input data.
        """
        query: str = Field(description="The question to lookup in the company FAQs.")

    @property
    def input_schema(self) -> type[BaseModel]:
        """Returns the input schema class for validation."""
        return self.FAQToolInput

    def _create_emitter(self) -> Emitter | None:
        """Creates an emitter for tracking tool execution events."""
        return Emitter()

    def __init__(self, embedding_model: SentenceTransformer, chroma_collection: chromadb.Collection):
        """
        Initialize the FAQ tool with required components.

        Args:
            embedding_model: Model for converting text to embeddings
            chroma_collection: ChromaDB collection containing FAQ data
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.chroma_collection = chroma_collection

    async def _run(self, query: str) -> str:
        """
        Main execution method for the FAQ tool.

        This method performs semantic search by:
        1. Converting the query to embeddings
        2. Searching ChromaDB for similar FAQ content
        3. Formatting and returning the results

        Args:
            query: The user's question to search for

        Returns:
            str: Formatted FAQ information or error message
        """
        # Start OpenTelemetry span for FAQ tool execution
        if _tracer:
            with _tracer.start_as_current_span("faq_tool_execution") as span:
                span.set_attribute("tool.name", self.name)
                span.set_attribute("tool.description", self.description)
                span.set_attribute("faq.query", query)
                span.set_attribute("faq.query_length", len(query))

                try:
                    # Step 1: Convert the user query to embeddings for semantic search
                    with _tracer.start_as_current_span("query_embedding") as embedding_span:
                        embedding_span.set_attribute("embedding.model", "sentence-transformers")
                        embedding_span.set_attribute("embedding.model_name", "all-MiniLM-L6-v2")

                        query_embedding = self.embedding_model.encode(query).tolist()
                        embedding_span.set_attribute("embedding.vector_size", len(query_embedding))
                        embedding_span.set_attribute("embedding.success", True)

                    # Step 2: Search ChromaDB for similar FAQ content using vector similarity
                    with _tracer.start_as_current_span("chroma_search") as search_span:
                        search_span.set_attribute("chroma.collection", CHROMA_COLLECTION_NAME)
                        search_span.set_attribute("chroma.n_results", 3)

                        results = self.chroma_collection.query(
                            query_embeddings=[query_embedding],  # Search using the query embedding
                            n_results=3,  # Return top 3 most similar results
                            include=['documents', 'metadatas']  # Include both content and metadata
                        )

                        search_span.set_attribute("chroma.results_count", len(results.get('documents', [[]])[0]) if results and results.get('documents') else 0)
                        search_span.set_attribute("chroma.search_success", True)

                    # Step 3: Process and format the search results
                    with _tracer.start_as_current_span("result_processing") as process_span:
                        retrieved_contexts = []
                        if results and results.get('documents') and results['documents'][0]:
                            # Iterate through the retrieved documents and format them
                            for i in range(len(results['documents'][0])):
                                doc_content = results['documents'][0][i]  # The FAQ content
                                metadata = results['metadatas'][0][i]     # Associated metadata
                                question = metadata.get('question', 'N/A')  # Extract question from metadata
                                answer = metadata.get('answer', doc_content)  # Extract answer or use content
                                retrieved_contexts.append(f"Question: {question}\nAnswer: {answer}")

                        process_span.set_attribute("processing.contexts_count", len(retrieved_contexts))
                        process_span.set_attribute("processing.success", True)

                    # Step 4: Handle case where no relevant information is found
                    if not retrieved_contexts:
                        span.set_attribute("faq.no_results", True)
                        span.set_attribute("faq.status", "no_results")
                        return "No relevant information found in the FAQs."

                    # Step 5: Combine all retrieved contexts into a single formatted string
                    context_string = "\n\n".join(retrieved_contexts)
                    span.set_attribute("faq.results_found", True)
                    span.set_attribute("faq.status", "success")
                    span.set_attribute("faq.response_length", len(context_string))

                    return context_string

                except Exception as e:
                    # Set span attributes for error
                    span.set_attribute("faq.status", "error")
                    span.set_attribute("error.message", str(e))
                    span.set_attribute("error.type", type(e).__name__)

                    # Log error for observability
                    print(f"Error in FAQ tool execution: {e}")
                    return f"Error processing query for FAQ lookup: {e}"
        else:
            # Fallback without OpenTelemetry
            try:
                # Step 1: Convert the user query to embeddings for semantic search
                query_embedding = self.embedding_model.encode(query).tolist()

                # Step 2: Search ChromaDB for similar FAQ content using vector similarity
                results = self.chroma_collection.query(
                    query_embeddings=[query_embedding],  # Search using the query embedding
                    n_results=3,  # Return top 3 most similar results
                    include=['documents', 'metadatas']  # Include both content and metadata
                )

                # Step 3: Process and format the search results
                retrieved_contexts = []
                if results and results.get('documents') and results['documents'][0]:
                    # Iterate through the retrieved documents and format them
                    for i in range(len(results['documents'][0])):
                        doc_content = results['documents'][0][i]  # The FAQ content
                        metadata = results['metadatas'][0][i]     # Associated metadata
                        question = metadata.get('question', 'N/A')  # Extract question from metadata
                        answer = metadata.get('answer', doc_content)  # Extract answer or use content
                        retrieved_contexts.append(f"Question: {question}\nAnswer: {answer}")

                # Step 4: Handle case where no relevant information is found
                if not retrieved_contexts:
                    return "No relevant information found in the FAQs."

                # Step 5: Combine all retrieved contexts into a single formatted string
                context_string = "\n\n".join(retrieved_contexts)
                return context_string

            except Exception as e:
                return f"Error processing query for FAQ lookup: {e}"

async def run_faq_agent(user_query: str) -> str:
    """
    Main function to run the FAQ agent workflow.

    This function orchestrates the entire FAQ answering process:
    1. Ensures the RAG system is properly initialized
    2. Searches for relevant FAQ information
    3. Generates a comprehensive answer using the LLM

    Args:
        user_query: The user's question about company FAQs

    Returns:
        str: The agent's response to the user's question
    """
    # Start OpenTelemetry span for the main FAQ agent workflow
    if _tracer:
        with _tracer.start_as_current_span("faq_agent_workflow") as span:
            span.set_attribute("workflow.name", "Company FAQ Assistant")
            span.set_attribute("workflow.query", user_query)
            span.set_attribute("workflow.query_length", len(user_query))
            span.set_attribute("workflow.timestamp", time.time())

            try:
                # Ensure the RAG system is set up before processing
                with _tracer.start_as_current_span("rag_system_setup") as setup_span:
                    if not _setup_rag_system():
                        setup_span.set_attribute("setup.status", "failed")
                        span.set_attribute("workflow.status", "setup_failed")
                        return "Backend RAG system failed to initialize. Please check server logs."
                    setup_span.set_attribute("setup.status", "success")

                # Step 1: Manually call the FAQ tool to retrieve relevant information
                print("Calling the faq_lookup_tool...")
                with _tracer.start_as_current_span("faq_tool_execution") as tool_span:
                    tool_span.set_attribute("tool.name", "faq_lookup_tool")
                    retrieved_info = await _faq_tool_instance._run(user_query)
                    tool_span.set_attribute("tool.response_length", len(retrieved_info))
                    tool_span.set_attribute("tool.success", True)

                # Step 2: Build a comprehensive prompt combining retrieved info and user question
                # This gives the LLM context about relevant FAQs before asking it to answer
                prompt_for_agent = f"Retrieved Company FAQ Information:\n{retrieved_info}\n\nUser Question: {user_query}"

                with _tracer.start_as_current_span("prompt_construction") as prompt_span:
                    prompt_span.set_attribute("prompt.retrieved_info_length", len(retrieved_info))
                    prompt_span.set_attribute("prompt.final_length", len(prompt_for_agent))
                    prompt_span.set_attribute("prompt.construction_success", True)

                # Step 3: Run the agent workflow to generate the final answer
                with _tracer.start_as_current_span("agent_workflow_execution") as workflow_span:
                    workflow_span.set_attribute("workflow.agent_name", "FAQAgent")
                    workflow_span.set_attribute("workflow.llm_model", "openai:gpt-4o")

                    response = await _agent_workflow.run(
                        inputs=[
                            AgentWorkflowInput(
                                prompt=prompt_for_agent,  # Send the combined prompt to the agent
                            )
                        ]
                    )

                    final_answer = response.result.final_answer
                    workflow_span.set_attribute("workflow.response_length", len(final_answer))
                    workflow_span.set_attribute("workflow.success", True)
                    span.set_attribute("workflow.status", "success")

                    return final_answer  # Extract the agent's response

            except Exception as e:
                # Set span attributes for error
                span.set_attribute("workflow.status", "error")
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)

                print(f"Error running agent workflow: {e}")
                return f"An error occurred while processing your request: {e}"
    else:
        # Fallback without OpenTelemetry
        # Ensure the RAG system is set up before processing
        if not _setup_rag_system():
            return "Backend RAG system failed to initialize. Please check server logs."

        # Step 1: Manually call the FAQ tool to retrieve relevant information
        print("Calling the faq_lookup_tool...")
        retrieved_info = await _faq_tool_instance._run(user_query)

        # Step 2: Build a comprehensive prompt combining retrieved info and user question
        # This gives the LLM context about relevant FAQs before asking it to answer
        prompt_for_agent = f"Retrieved Company FAQ Information:\n{retrieved_info}\n\nUser Question: {user_query}"

        # Step 3: Run the agent workflow to generate the final answer
        try:
            response = await _agent_workflow.run(
                inputs=[
                    AgentWorkflowInput(
                        prompt=prompt_for_agent,  # Send the combined prompt to the agent
                    )
                ]
            )
            return response.result.final_answer  # Extract the agent's response
        except Exception as e:
            print(f"Error running agent workflow: {e}")
            return f"An error occurred while processing your request: {e}"

def get_observability_data() -> dict:
    """
    Get observability data for monitoring the RAG system.

    Returns:
        dict: Dictionary containing observability metrics and status
    """
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