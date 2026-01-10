# backend/main.py
import os
import uuid

# Load environment variables FIRST before importing modules that need them
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from starlette.middleware.cors import CORSMiddleware
from openinference.instrumentation.beeai import BeeAIInstrumentor
from contextlib import asynccontextmanager

# Import OpenTelemetry for manual span tagging
from opentelemetry import trace

# Import existing modules
from backend.agent import run_faq_agent, _setup_rag_system

# Import new sentiment analysis modules
from backend.sentiment_analyzer import SentimentAnalyzer, ConversationTracker, generate_conversation_summary
from backend.email_service import EmailService

# Initialize sentiment analysis components
sentiment_analyzer = SentimentAnalyzer(frustration_threshold=0.6)
conversation_tracker = ConversationTracker(frustration_threshold=0.6, trigger_count=3)
email_service = EmailService()

# Simple global counter for frustrated messages
global_frustrated_count = 0
global_conversation_history = []
escalation_triggered = False
waiting_for_representative = False
user_responded_to_escalation = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan context manager for FastAPI startup/shutdown events."""
    # Startup
    print("FastAPI startup event: Initializing RAG system...")
    success = _setup_rag_system()
    if not success:
        print("RAG system initialization failed during startup.")
    else:
        print("RAG system initialized successfully.")
    
    # Test email service connection
    print("\nTesting email service configuration...")
    email_service.test_connection()
    
    BeeAIInstrumentor().instrument()
    print("\n✅ Sentiment Analysis and Escalation System Enabled")
    print(f"   Frustration Threshold: {sentiment_analyzer.frustration_threshold}")
    print(f"   Escalation Trigger: {conversation_tracker.trigger_count} frustrated messages")
    
    yield
    
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(
    title="Company FAQ RAG API",
    description="Backend API for the Company FAQ Retrieval Augmented Generation (RAG) system with Sentiment Analysis.",
    version="2.0.0",
    lifespan=lifespan
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8080",
    "http://0.0.0.0:8000",
    "http://localhost:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory="frontend/static"),
    name="static"
)

templates = Jinja2Templates(directory="frontend")

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sentiment_score: float
    frustrated_count: int
    should_escalate: bool
    escalation_message: Optional[str] = None
    session_id: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main HTML page for the frontend."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request_body: ChatRequest):
    """
    Enhanced chat endpoint with sentiment analysis and escalation detection.
    Uses simple global counter to track frustrated messages.
    """
    global global_frustrated_count, global_conversation_history, escalation_triggered, waiting_for_representative, user_responded_to_escalation
    
    user_query = request_body.query
    session_id = request_body.session_id
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # --- OPENTELEMETRY FIX: Tag the current span with session_id ---
    current_span = trace.get_current_span()
    if current_span:  # Check if a span is active (it should be, via auto-instrumentation)
        current_span.set_attribute("session.id", session_id)
        current_span.set_attribute("session_id", session_id)  # Alternative format for compatibility
        print(f"✅ Session ID tagged in OpenTelemetry span: {session_id[:8]}...")
    # --- End Fix ---
    
    print(f"\n{'='*60}")
    print(f"📨 Received query")
    print(f"   Session ID: {session_id[:8]}...")
    print(f"   Query: {user_query}")
    
    # Check if user is responding to escalation notice (ONLY ONCE after notification)
    if escalation_triggered and not user_responded_to_escalation:
        user_query_lower = user_query.lower()
        
        # Keywords indicating user wants to wait for representative
        wait_keywords = ['wait', 'representative', 'rep', 'human', 'person', 'agent', 'support', 'team']
        # Keywords indicating user wants to continue chatting
        continue_keywords = ['continue', 'chat', 'keep going', 'go on', 'help me', 'no']
        
        wants_to_wait = any(keyword in user_query_lower for keyword in wait_keywords)
        wants_to_continue = any(keyword in user_query_lower for keyword in continue_keywords)
        
        if wants_to_wait:
            user_responded_to_escalation = True
            waiting_for_representative = True
            agent_answer = (
                "Thank you for your patience. A support representative has been notified and will contact you shortly. "
                "You should receive a follow-up within the next hour. Is there anything else I can note for the representative?"
            )
            print("✅ User chose to wait for representative")
            print("🔒 User response to escalation recorded - keyword detection disabled")
            
            return ChatResponse(
                answer=agent_answer,
                sentiment_score=0.0,
                frustrated_count=global_frustrated_count,
                should_escalate=False,
                escalation_message=None,
                session_id=session_id
            )
        
        elif wants_to_continue:
            user_responded_to_escalation = True
            print("✅ User chose to continue chatting")
            print("🔒 User response to escalation recorded - keyword detection disabled")
    
    # If user is waiting for representative, give holding response
    if waiting_for_representative:
        agent_answer = (
            "I understand you're waiting for a representative. They have been notified and will reach out soon. "
            "I've passed along your message. Is there anything else you'd like me to add to the notes for them?"
        )
        print("ℹ️  User is waiting for representative - providing holding response")
        
        return ChatResponse(
            answer=agent_answer,
            sentiment_score=0.0,
            frustrated_count=global_frustrated_count,
            should_escalate=False,
            escalation_message=None,
            session_id=session_id
        )
    
    try:
        # Step 1: Get response from RAG agent
        agent_answer = await run_faq_agent(user_query)
        print(f"Agent response generated ({len(agent_answer)} chars)")
        
        # Step 2: Analyze sentiment of user message
        sentiment_score = await sentiment_analyzer.analyze_sentiment(user_query)
        print(f"Sentiment score: {sentiment_score:.2f}")
        
        # Tag sentiment score in OpenTelemetry span
        if current_span:
            current_span.set_attribute("sentiment.score", sentiment_score)
        
        # Categorize sentiment for logging
        if sentiment_score < 0.3:
            sentiment_label = "😊 Neutral/Positive"
            sentiment_category = "neutral_positive"
        elif sentiment_score < 0.6:
            sentiment_label = "😐 Slightly Frustrated"
            sentiment_category = "slightly_frustrated"
        elif sentiment_score < 0.8:
            sentiment_label = "😠 Moderately Frustrated"
            sentiment_category = "moderately_frustrated"
        else:
            sentiment_label = "😡 Highly Frustrated"
            sentiment_category = "highly_frustrated"
        
        print(f"Sentiment: {sentiment_label}")
        
        # Tag sentiment category in OpenTelemetry span
        if current_span:
            current_span.set_attribute("sentiment.category", sentiment_category)
            current_span.set_attribute("sentiment.label", sentiment_label)
        
        # Step 3: Simple counter - increment if frustrated
        if sentiment_score >= 0.6:
            global_frustrated_count += 1
            print(f"✅ Frustrated message detected! Count: {global_frustrated_count}/3")
        else:
            print(f"📊 Not frustrated. Count remains: {global_frustrated_count}/3")
        
        # Tag frustrated count in OpenTelemetry span
        if current_span:
            current_span.set_attribute("frustrated.count", global_frustrated_count)
            current_span.set_attribute("frustrated.threshold", 3)
            current_span.set_attribute("frustrated.detected", sentiment_score >= 0.6)
        
        # Store message in history
        global_conversation_history.append({
            "user_message": user_query,
            "bot_response": agent_answer,
            "sentiment_score": sentiment_score
        })
        
        escalation_message = None
        should_escalate = False
        
        # Step 4: Check if we should escalate
        if global_frustrated_count >= 3 and not escalation_triggered:
            should_escalate = True
            escalation_triggered = True
            print("\n🚨 ESCALATION TRIGGERED!")
            
            # Tag escalation in OpenTelemetry span
            if current_span:
                current_span.set_attribute("escalation.triggered", True)
                current_span.set_attribute("escalation.session_id", session_id)
            
            try:
                # Generate summary from conversation history
                summary_text = f"""ESCALATED CONVERSATION SUMMARY
{'='*50}

CONVERSATION HISTORY:
"""
                for i, msg in enumerate(global_conversation_history, 1):
                    summary_text += f"\nMessage {i} (Score: {msg['sentiment_score']:.2f}):\n"
                    summary_text += f"User: {msg['user_message']}\n"
                    summary_text += f"Bot: {msg['bot_response']}\n"
                
                summary_text += f"""
FRUSTRATION METRICS:
- Total Frustrated Messages: {global_frustrated_count}
- Total Messages: {len(global_conversation_history)}
- Average Sentiment Score: {sum(m['sentiment_score'] for m in global_conversation_history) / len(global_conversation_history):.2f}
"""
                
                print("\n📧 Attempting to send escalation email...")
                email_sent = email_service.send_escalation_email(
                    conversation_summary=summary_text,
                    session_id=session_id
                )
                
                # Tag email status in OpenTelemetry span
                if current_span:
                    current_span.set_attribute("email.sent", email_sent)
                    current_span.set_attribute("email.recipient", email_service.recipient_email or "not_configured")
                
                if email_sent:
                    print("✅ Escalation email sent successfully!")
                else:
                    print("⚠️ Email failed to send - check configuration")
                
            except Exception as email_error:
                print(f"❌ Error during escalation process: {email_error}")
            
            escalation_message = (
                "I notice you may be experiencing some frustration. "
                "I've notified our support team, and a representative will reach out to you shortly. "
                "Would you like to continue chatting with me, or would you prefer to wait for a representative?"
            )
            
            print(f"\n📧 Escalation Process Complete")
            print(f"🔓 Keyword detection enabled for next user message")
            print(f"{'='*60}\n")
        
        # Step 5: Return response
        return ChatResponse(
            answer=agent_answer,
            sentiment_score=sentiment_score,
            frustrated_count=global_frustrated_count,
            should_escalate=should_escalate,
            escalation_message=escalation_message,
            session_id=session_id
        )
        
    except Exception as e:
        print(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history."""
    return {
        "frustrated_count": global_frustrated_count,
        "total_messages": len(global_conversation_history),
        "escalation_triggered": escalation_triggered,
        "messages": global_conversation_history
    }

@app.post("/session/reset")
async def reset_session():
    """Reset the global frustrated counter and conversation history."""
    global global_frustrated_count, global_conversation_history, escalation_triggered, waiting_for_representative, user_responded_to_escalation
    
    global_frustrated_count = 0
    global_conversation_history = []
    escalation_triggered = False
    waiting_for_representative = False
    user_responded_to_escalation = False
    
    return {"message": "Session reset successfully", "frustrated_count": 0}

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "rag_system": "operational",
        "sentiment_analysis": "enabled",
        "email_service": "configured" if email_service.recipient_email else "not_configured",
        "frustrated_count": global_frustrated_count,
        "total_messages": len(global_conversation_history),
        "escalation_triggered": escalation_triggered
    }

if __name__ == "__main__":
    import uvicorn
    port = 8001
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)