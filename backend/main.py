import os
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

# Import live agent system components
from models import (
    ChatRequest, ChatResponse, SessionState, ChatMessage,
    HandoffRequest, SessionInfo, AgentInfo
)
from live_agent_system import session_manager
from websocket_manager import connection_manager

# Observability Import with Graceful Degradation
try:
    from openinference.instrumentation.beeai import BeeAIInstrumentor
except ImportError:
    BeeAIInstrumentor = None

from agent import run_faq_agent, _setup_rag_system
from sentiment_analyzer import SentimentAnalyzer, ConversationTracker, generate_conversation_summary
from email_service import EmailService

# Initialize sentiment analysis and email service (singletons shared across requests)
_sentiment_analyzer = SentimentAnalyzer(frustration_threshold=0.6)
_conversation_tracker = ConversationTracker(frustration_threshold=0.6, trigger_count=3)
_email_service = EmailService()

load_dotenv()

app = FastAPI(
    title="Company FAQ RAG API with Live Agent Handoff",
    description="Backend API for the Company FAQ Retrieval Augmented Generation (RAG) system with live agent support.",
    version="2.0.0",
)


def _parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost,http://localhost:8000,http://127.0.0.1:8080,http://0.0.0.0:8000,http://localhost:8001,http://127.0.0.1:8001",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


AGENT_API_KEY = os.getenv("AGENT_API_KEY", "").strip()
origins = _parse_cors_origins()


def _enforce_agent_api_key(request: Request = None, websocket: WebSocket = None):
    """Require agent API key only when AGENT_API_KEY is configured."""
    if not AGENT_API_KEY:
        return

    provided = ""
    if request is not None:
        provided = request.headers.get("x-agent-api-key", "").strip()
        if provided != AGENT_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid agent API key")
    elif websocket is not None:
        provided = websocket.query_params.get("api_key", "").strip()
        if provided != AGENT_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid agent API key")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Get the base directory (works both locally and in Docker)
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "frontend" / "static"
TEMPLATES_DIR = BASE_DIR / "frontend"

# Ensure directories exist
if not STATIC_DIR.exists():
    print(f"Warning: Static directory not found at {STATIC_DIR}")
if not TEMPLATES_DIR.exists():
    print(f"Warning: Templates directory not found at {TEMPLATES_DIR}")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    from agent import get_observability_data
    obs_data = get_observability_data()
    return {
        "status": "healthy" if obs_data.get("status") == "ready" else "initializing",
        "service": "ai-observability-bot",
        "rag_system": obs_data.get("rag_system", {}),
        "chroma_db": obs_data.get("chroma_db", {})
    }


AGENT_WAIT_TIMEOUT_SECONDS = int(os.getenv("AGENT_WAIT_TIMEOUT_SECONDS", "60"))
AGENT_WAIT_CHECK_INTERVAL = 10  # check every 10 s

TIMEOUT_MESSAGE = (
    "We apologize, all of our agents are assisting other customers right now. "
    "Please try again in a few minutes."
)

async def _agent_wait_timeout_task():
    """Background task: cancel sessions that waited too long for a live agent."""
    while True:
        await asyncio.sleep(AGENT_WAIT_CHECK_INTERVAL)
        try:
            timed_out = session_manager.get_timed_out_sessions(AGENT_WAIT_TIMEOUT_SECONDS)
            for session in timed_out:
                sid = session.session_id
                print(f"[WaitTimeout] Session {sid} waited >{AGENT_WAIT_TIMEOUT_SECONDS}s — returning to AI mode")

                # Cancel the handoff (sets state back to AI, removes from queue)
                session_manager.cancel_handoff(sid)
                session_manager.add_message(sid, ChatMessage(
                    sender="system",
                    content=TIMEOUT_MESSAGE
                ))

                # Try to push the message over WebSocket
                if connection_manager.is_customer_connected(sid):
                    await connection_manager.send_to_customer(sid, {
                        "type": "timeout",
                        "message": TIMEOUT_MESSAGE
                    })
                    print(f"[WaitTimeout] Timeout message sent via WebSocket to {sid}")
                else:
                    # WS already closed — store for delivery on the next HTTP request
                    s = session_manager.get_session(sid)
                    if s:
                        s.pending_timeout_msg = TIMEOUT_MESSAGE
                    print(f"[WaitTimeout] WS not connected for {sid} — stored as pending HTTP message")
        except Exception as e:
            print(f"[WaitTimeout] Error in timeout task: {e}")


@app.on_event("startup")
async def startup_event():
    """Initializes the RAG system when the FastAPI app starts up."""
    print("FastAPI startup event: Initializing RAG system...")
    success = _setup_rag_system()
    if not success:
        print("RAG system initialization failed during startup.")
    else:
        print("RAG system initialized successfully.")

    # Initialize BeeAI Instrumentor if available
    if BeeAIInstrumentor:
        try:
            BeeAIInstrumentor().instrument()
            print("BeeAI Instrumentor initialized.")
        except Exception as e:
            print(f"Failed to initialize BeeAI Instrumentor: {e}")

    # Start background task that auto-cancels timed-out agent waits
    asyncio.create_task(_agent_wait_timeout_task())
    print(f"Agent wait timeout task started (timeout={AGENT_WAIT_TIMEOUT_SECONDS}s, check every {AGENT_WAIT_CHECK_INTERVAL}s)")


# ============================================================================
# CUSTOMER ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main HTML page for the frontend."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/agent-dashboard", response_class=HTMLResponse)
@app.get("/agent-dashboard.html", response_class=HTMLResponse)
async def serve_agent_dashboard(request: Request):
    """Serve the agent dashboard HTML page."""
    return templates.TemplateResponse("agent-dashboard.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request_body: ChatRequest):
    """
    Main chat endpoint that handles both AI and live agent routing.
    Detects handoff keywords and manages session state.
    """
    user_query = request_body.query
    # tracker_session_id always stays equal to what the frontend sent (or the first-time UUID).
    # This keeps the frustrated_count accumulation stable even if the live-agent session_manager
    # loses state (e.g. after a container restart) and has to create a new internal session.
    frontend_session_id = request_body.session_id

    print(f"Received query: {user_query}")

    # Create or get session for live-agent state tracking
    if not frontend_session_id:
        session_id = session_manager.create_session()
        print(f"Created new session: {session_id}")
    else:
        session = session_manager.get_session(frontend_session_id)
        if session:
            session_id = frontend_session_id
        else:
            # session_manager lost state (restart); create a new live-agent session
            # but keep the original id for the sentiment tracker below
            print(f"Session {frontend_session_id} not found in session_manager, creating new live-agent session")
            session_id = session_manager.create_session()

    # tracker always uses the id the frontend knows about so frustrated_count is stable
    tracker_session_id = frontend_session_id or session_id
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to create session")
    
    # Add customer message to history
    session_manager.add_message(session_id, ChatMessage(
        sender="customer",
        content=user_query
    ))

    # HTTP fallback: deliver any timeout message the WebSocket couldn't deliver
    pending_session = session_manager.get_session(session_id)
    if pending_session and pending_session.pending_timeout_msg:
        msg = pending_session.pending_timeout_msg
        pending_session.pending_timeout_msg = None   # clear after delivering
        return ChatResponse(
            answer=msg,
            session_id=session_id,
            state=SessionState.AI
        )

    # Check if customer wants to talk to a live agent
    if session_manager.detect_handoff_request(user_query):
        print(f"Handoff request detected for session {session_id}")

        # Check how many agents are currently connected via WebSocket
        online_count = session_manager.get_online_agent_count(
            set(connection_manager.agent_connections.keys())
        )

        if online_count == 0:
            # No agents online — stay in AI mode, don't enter waiting queue
            no_agent_msg = (
                "There's currently no live agent available. "
                "Please try again later or continue chatting with me — I'm happy to help!"
            )
            session_manager.add_message(session_id, ChatMessage(
                sender="system",
                content=no_agent_msg
            ))
            return ChatResponse(
                answer=no_agent_msg,
                session_id=session_id,
                state=SessionState.AI
            )

        session_manager.request_handoff(session_id)

        # Notify all connected agents about new customer in queue
        await connection_manager.broadcast_to_all_agents({
            "type": "new_customer",
            "session_id": session_id,
            "message": "New customer waiting in queue"
        })

        response_text = (
            "I'll connect you with a live agent right away. "
            "Please wait a moment while I find someone to help you... "
            "Type 'cancel' at any time to go back to the AI assistant."
        )

        session_manager.add_message(session_id, ChatMessage(
            sender="system",
            content=response_text
        ))

        return ChatResponse(
            answer=response_text,
            session_id=session_id,
            state=SessionState.WAITING_FOR_AGENT
        )

    # If session is in live agent mode, don't process with AI
    if session.state == SessionState.LIVE_AGENT:
        return ChatResponse(
            answer="You are currently connected to a live agent. Please use the chat to communicate.",
            session_id=session_id,
            state=session.state,
            agent_name=session.agent_name
        )

    # If session is waiting for agent, allow cancel or remind them
    if session.state == SessionState.WAITING_FOR_AGENT:
        if session_manager.detect_cancel_request(user_query):
            session_manager.cancel_handoff(session_id)
            cancel_msg = (
                "No problem! I've cancelled the live agent request. "
                "I'm here to help — what would you like to know?"
            )
            session_manager.add_message(session_id, ChatMessage(
                sender="system",
                content=cancel_msg
            ))
            return ChatResponse(
                answer=cancel_msg,
                session_id=session_id,
                state=SessionState.AI
            )

        return ChatResponse(
            answer=(
                "Still looking for an available agent, please hang tight... "
                "Type 'cancel' if you'd like to go back to the AI assistant."
            ),
            session_id=session_id,
            state=session.state
        )
    
    # Process with AI agent
    try:
        agent_answer = await run_faq_agent(user_query)

        session_manager.add_message(session_id, ChatMessage(
            sender="agent",
            content=agent_answer
        ))

        # --- Sentiment analysis & escalation ---
        try:
            sentiment_score = await _sentiment_analyzer.analyze_sentiment(user_query)
            tracking_result = _conversation_tracker.track_message(
                tracker_session_id, user_query, agent_answer, sentiment_score
            )
            print(
                f"[Sentiment] session={tracker_session_id} score={sentiment_score:.2f} "
                f"frustrated_count={tracking_result['frustrated_count']}"
            )

            if tracking_result["should_escalate"]:
                print(f"[Escalation] Triggering escalation for session {tracker_session_id}")
                history = _conversation_tracker.get_conversation_history(tracker_session_id)
                summary = await generate_conversation_summary(history)
                email_sent = _email_service.send_escalation_email(summary, session_id)
                if not email_sent:
                    print(f"[Escalation] Email failed for session {session_id} — see above for details")
        except Exception as sentiment_err:
            # Sentiment errors must never break the chat response
            print(f"[Sentiment] Error (non-fatal): {sentiment_err}")
        # --- end sentiment ---

        return ChatResponse(
            answer=agent_answer,
            session_id=session_id,
            state=session.state
        )
    except Exception as e:
        print(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


WEBSOCKET_PING_INTERVAL = 20  # seconds between server-side pings

@app.websocket("/ws/customer/{session_id}")
async def customer_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for customer real-time chat."""
    await connection_manager.connect_customer(session_id, websocket)
    session_manager.set_customer_connected(session_id, True)

    try:
        while True:
            # Wait for a message; if none arrives within PING_INTERVAL send a ping
            # to keep the browser from closing the idle connection.
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=WEBSOCKET_PING_INTERVAL
                )
            except asyncio.TimeoutError:
                # No message received — send a lightweight ping to keep alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break   # WebSocket closed; exit loop cleanly
                continue
            message_type = data.get("type")
            
            if message_type == "message":
                content = data.get("content", "")
                session = session_manager.get_session(session_id)
                
                if not session:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not found"
                    })
                    break
                
                # Add to history
                session_manager.add_message(session_id, ChatMessage(
                    sender="customer",
                    content=content
                ))
                
                # If in live agent mode, forward to agent
                if session.state == SessionState.LIVE_AGENT and session.agent_id:
                    await connection_manager.send_to_agent(session.agent_id, {
                        "type": "customer_message",
                        "session_id": session_id,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif message_type == "typing":
                # Forward typing indicator to agent
                session = session_manager.get_session(session_id)
                if session and session.state == SessionState.LIVE_AGENT and session.agent_id:
                    await connection_manager.send_to_agent(session.agent_id, {
                        "type": "customer_typing",
                        "session_id": session_id,
                        "is_typing": data.get("is_typing", False)
                    })
    
    except WebSocketDisconnect:
        print(f"Customer WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"Error in customer WebSocket: {e}")
    finally:
        connection_manager.disconnect_customer(session_id)
        session_manager.set_customer_connected(session_id, False)


# ============================================================================
# AGENT DASHBOARD ENDPOINTS
# ============================================================================

@app.get("/agent-dashboard", response_class=HTMLResponse)
async def serve_agent_dashboard(request: Request):
    """Serve the agent dashboard HTML page."""
    return templates.TemplateResponse("agent-dashboard.html", {"request": request})


@app.post("/api/agent/login")
async def agent_login(agent_id: str, name: str, request: Request):
    """Register/login an agent."""
    _enforce_agent_api_key(request=request)
    agent = session_manager.register_agent(agent_id, name)
    return {
        "success": True,
        "agent": {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "is_available": agent.is_available
        }
    }


@app.get("/api/agent/queue")
async def get_waiting_queue(request: Request):
    """Get list of customers waiting for an agent."""
    _enforce_agent_api_key(request=request)
    waiting_sessions = session_manager.get_waiting_sessions()
    return {
        "queue": [
            {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "message_count": len(session.messages),
                "last_message": session.messages[-1].content if session.messages else None
            }
            for session in waiting_sessions
        ]
    }


@app.post("/api/agent/accept/{session_id}")
async def accept_chat(session_id: str, agent_id: str, request: Request):
    """Agent accepts a customer chat."""
    _enforce_agent_api_key(request=request)
    success = session_manager.assign_agent(session_id, agent_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign agent")
    
    session = session_manager.get_session(session_id)
    
    # Notify customer via WebSocket
    await connection_manager.send_to_customer(session_id, {
        "type": "agent_joined",
        "agent_name": session.agent_name,
        "message": f"{session.agent_name} has joined the chat"
    })
    
    return {
        "success": True,
        "session": {
            "session_id": session.session_id,
            "state": session.state,
            "agent_name": session.agent_name,
            "messages": [
                {
                    "sender": msg.sender,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "agent_name": msg.agent_name
                }
                for msg in session.messages
            ]
        }
    }


@app.post("/api/agent/end/{session_id}")
async def end_agent_session_endpoint(session_id: str, return_to_ai: bool = False, request: Request = None):
    """End an agent session."""
    _enforce_agent_api_key(request=request)
    success = session_manager.end_agent_session(session_id, return_to_ai)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to end session")
    
    # Notify customer
    await connection_manager.send_to_customer(session_id, {
        "type": "agent_left",
        "return_to_ai": return_to_ai,
        "message": "Agent has left the chat" if not return_to_ai else "Agent has left. You can continue with AI assistant."
    })
    
    return {"success": True}


@app.get("/api/agent/sessions/{agent_id}")
async def get_agent_sessions(agent_id: str, request: Request):
    """Get all active sessions for an agent."""
    _enforce_agent_api_key(request=request)
    sessions = session_manager.get_agent_sessions(agent_id)
    return {
        "sessions": [
            {
                "session_id": session.session_id,
                "state": session.state,
                "created_at": session.created_at.isoformat(),
                "message_count": len(session.messages)
            }
            for session in sessions
        ]
    }


@app.websocket("/ws/agent/{agent_id}")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for agent real-time communication."""
    try:
        _enforce_agent_api_key(websocket=websocket)
    except HTTPException:
        await websocket.close(code=1008)
        return

    await connection_manager.connect_agent(agent_id, websocket)
    
    try:
        while True:
            # Receive message from agent
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "message":
                session_id = data.get("session_id")
                content = data.get("content", "")
                
                session = session_manager.get_session(session_id)
                if not session:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Session not found"
                    })
                    continue
                
                # Add to history
                session_manager.add_message(session_id, ChatMessage(
                    sender="agent",
                    content=content,
                    agent_name=session.agent_name
                ))
                
                # Forward to customer
                await connection_manager.send_to_customer(session_id, {
                    "type": "agent_message",
                    "content": content,
                    "agent_name": session.agent_name,
                    "timestamp": datetime.now().isoformat()
                })
            
            elif message_type == "typing":
                # Forward typing indicator to customer
                session_id = data.get("session_id")
                await connection_manager.send_to_customer(session_id, {
                    "type": "agent_typing",
                    "is_typing": data.get("is_typing", False)
                })
    
    except WebSocketDisconnect:
        print(f"Agent WebSocket disconnected: {agent_id}")
    except Exception as e:
        print(f"Error in agent WebSocket: {e}")
    finally:
        connection_manager.disconnect_agent(agent_id)


if __name__ == "__main__":
    import uvicorn
    port = 8001
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
