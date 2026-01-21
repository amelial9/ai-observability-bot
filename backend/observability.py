# backend/observability.py
"""
Comprehensive OpenTelemetry Observability Module for RAG Pipeline
Provides tracing, metrics, and logging for the entire RAG system.
"""

import os
import time
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.semconv.trace import SpanAttributes

# Configuration
OTEL_ENDPOINT = os.getenv("OTEL_ENDPOINT", "http://localhost:4328")
SERVICE_NAME = os.getenv("SERVICE_NAME", "rag-faq-system")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Global providers
_tracer_provider: Optional[TracerProvider] = None
_meter_provider: Optional[MeterProvider] = None
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None

# Metrics
_rag_query_counter = None
_rag_duration_histogram = None
_embedding_duration_histogram = None
_llm_token_counter = None
_sentiment_score_histogram = None
_escalation_counter = None
_cache_hit_counter = None


def setup_observability(
    service_name: str = SERVICE_NAME,
    otel_endpoint: str = OTEL_ENDPOINT,
    environment: str = ENVIRONMENT
) -> tuple[trace.Tracer, metrics.Meter]:
    """
    Initialize OpenTelemetry tracing and metrics.

    Args:
        service_name: Name of the service for identification
        otel_endpoint: OTLP endpoint URL (without /v1/traces or /v1/metrics)
        environment: Deployment environment (dev, staging, production)

    Returns:
        Tuple of (tracer, meter) for creating spans and metrics
    """
    global _tracer_provider, _meter_provider, _tracer, _meter
    global _rag_query_counter, _rag_duration_histogram, _embedding_duration_histogram
    global _llm_token_counter, _sentiment_score_histogram, _escalation_counter, _cache_hit_counter

    # Check if already initialized
    if _tracer and _meter:
        print("✅ OpenTelemetry already initialized")
        return _tracer, _meter

    print(f"🔧 Initializing OpenTelemetry for {service_name}...")

    # Create resource with service information
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "2.0.0",
        "deployment.environment": environment,
        "telemetry.sdk.name": "opentelemetry",
        "telemetry.sdk.language": "python",
        "telemetry.sdk.version": "1.20.0"
    })

    # === TRACING SETUP ===
    _tracer_provider = TracerProvider(resource=resource)

    # Configure trace exporter
    trace_exporter = OTLPSpanExporter(
        endpoint=f"{otel_endpoint}/v1/traces",
        headers={}
    )

    # Add batch processor for efficient span export
    _tracer_provider.add_span_processor(
        BatchSpanProcessor(
            trace_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            schedule_delay_millis=5000
        )
    )

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    _tracer = trace.get_tracer(service_name, "1.0.0")

    print(f"✅ Tracing configured: {otel_endpoint}/v1/traces")

    # === METRICS SETUP ===
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=f"{otel_endpoint}/v1/metrics",
            headers={}
        ),
        export_interval_millis=10000  # Export every 10 seconds
    )

    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[metric_reader]
    )

    metrics.set_meter_provider(_meter_provider)
    _meter = metrics.get_meter(service_name, "1.0.0")

    print(f"✅ Metrics configured: {otel_endpoint}/v1/metrics")

    # === CREATE METRICS ===

    # Counter: Total RAG queries
    _rag_query_counter = _meter.create_counter(
        name="rag.queries.total",
        description="Total number of RAG queries processed",
        unit="1"
    )

    # Histogram: RAG query duration
    _rag_duration_histogram = _meter.create_histogram(
        name="rag.query.duration",
        description="Duration of RAG query processing",
        unit="ms"
    )

    # Histogram: Embedding generation duration
    _embedding_duration_histogram = _meter.create_histogram(
        name="rag.embedding.duration",
        description="Duration of embedding generation",
        unit="ms"
    )

    # Counter: LLM token usage
    _llm_token_counter = _meter.create_counter(
        name="llm.tokens.used",
        description="Total number of LLM tokens used",
        unit="tokens"
    )

    # Histogram: Sentiment scores
    _sentiment_score_histogram = _meter.create_histogram(
        name="sentiment.score",
        description="Distribution of sentiment scores",
        unit="1"
    )

    # Counter: Escalations triggered
    _escalation_counter = _meter.create_counter(
        name="escalation.triggered",
        description="Number of escalations triggered",
        unit="1"
    )

    # Counter: Cache hits/misses
    _cache_hit_counter = _meter.create_counter(
        name="rag.cache.hits",
        description="Number of cache hits vs misses",
        unit="1"
    )

    print("✅ Metrics instruments created")
    print(f"📊 OpenTelemetry fully initialized for {service_name}")

    return _tracer, _meter


def get_tracer() -> Optional[trace.Tracer]:
    """Get the global tracer instance."""
    return _tracer


def get_meter() -> Optional[metrics.Meter]:
    """Get the global meter instance."""
    return _meter


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True
):
    """
    Context manager for tracing operations with automatic error handling.

    Usage:
        with trace_operation("embedding_generation", {"model": "all-MiniLM-L6-v2"}):
            embeddings = model.encode(text)
    """
    if not _tracer:
        # No tracer available, just execute without tracing
        yield None
        return

    with _tracer.start_as_current_span(operation_name) as span:
        # Set initial attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        start_time = time.time()

        try:
            yield span

            # Set success status
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Record exception details
            if record_exception:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error", True)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))

            raise  # Re-raise the exception

        finally:
            # Record duration
            duration_ms = (time.time() - start_time) * 1000
            span.set_attribute("duration_ms", duration_ms)


def trace_rag_query(func: Callable) -> Callable:
    """
    Decorator for tracing RAG query functions.

    Usage:
        @trace_rag_query
        async def run_faq_agent(query: str) -> str:
            ...
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        if not _tracer:
            return await func(*args, **kwargs)

        # Extract query from args/kwargs
        query = kwargs.get('query') or (args[0] if args else "unknown")

        with _tracer.start_as_current_span("rag_query") as span:
            span.set_attribute("rag.query", str(query)[:500])  # Limit size
            span.set_attribute("rag.query_length", len(str(query)))

            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Record metrics
                duration_ms = (time.time() - start_time) * 1000
                if _rag_query_counter:
                    _rag_query_counter.add(1, {"status": "success"})
                if _rag_duration_histogram:
                    _rag_duration_histogram.record(duration_ms, {"status": "success"})

                span.set_attribute("rag.response_length", len(str(result)))
                span.set_attribute("duration_ms", duration_ms)
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                if _rag_query_counter:
                    _rag_query_counter.add(1, {"status": "error"})
                if _rag_duration_histogram:
                    _rag_duration_histogram.record(duration_ms, {"status": "error"})

                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    return wrapper


def record_embedding_operation(duration_ms: float, model_name: str, vector_size: int):
    """Record metrics for embedding operations."""
    if _embedding_duration_histogram:
        _embedding_duration_histogram.record(
            duration_ms,
            {"model": model_name, "vector_size": str(vector_size)}
        )


def record_llm_tokens(prompt_tokens: int, completion_tokens: int, model: str):
    """Record LLM token usage."""
    if _llm_token_counter:
        _llm_token_counter.add(
            prompt_tokens,
            {"token_type": "prompt", "model": model}
        )
        _llm_token_counter.add(
            completion_tokens,
            {"token_type": "completion", "model": model}
        )


def record_sentiment_score(score: float, category: str):
    """Record sentiment analysis score."""
    if _sentiment_score_histogram:
        _sentiment_score_histogram.record(score, {"category": category})


def record_escalation(session_id: str, frustrated_count: int):
    """Record escalation event."""
    if _escalation_counter:
        _escalation_counter.add(
            1,
            {
                "session_id": session_id[:8],
                "frustrated_count": str(frustrated_count)
            }
        )


def record_cache_operation(hit: bool, cache_type: str = "default"):
    """Record cache hit/miss."""
    if _cache_hit_counter:
        _cache_hit_counter.add(
            1,
            {
                "cache_type": cache_type,
                "result": "hit" if hit else "miss"
            }
        )


def add_span_attributes(attributes: Dict[str, Any]):
    """Add attributes to the current active span."""
    current_span = trace.get_current_span()
    if current_span:
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def shutdown_observability():
    """Gracefully shutdown OpenTelemetry providers."""
    print("🛑 Shutting down OpenTelemetry...")

    if _tracer_provider:
        _tracer_provider.force_flush(timeout_millis=5000)
        _tracer_provider.shutdown()

    if _meter_provider:
        _meter_provider.force_flush(timeout_millis=5000)
        _meter_provider.shutdown()

    print("✅ OpenTelemetry shutdown complete")


# Example usage patterns
"""
# 1. Initialize at startup
tracer, meter = setup_observability(
    service_name="rag-faq-system",
    otel_endpoint="http://localhost:4328",
    environment="production"
)

# 2. Use context manager for operations
with trace_operation("vector_search", {"collection": "faqs", "k": 3}):
    results = collection.query(embeddings)

# 3. Use decorator for functions
@trace_rag_query
async def process_query(query: str):
    return await rag_system.query(query)

# 4. Record metrics manually
record_sentiment_score(0.75, "moderately_frustrated")
record_llm_tokens(150, 50, "gpt-4o")

# 5. Add attributes to current span
add_span_attributes({
    "user.session_id": session_id,
    "retrieval.num_results": 3
})
"""
