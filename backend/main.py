import os
import uuid
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

load_dotenv()

app = FastAPI(
    title="Company FAQ RAG API with Live Agent Handoff",
    description="Backend API for the Company FAQ Retrieval Augmented Generation (RAG) system with live agent support.",
    version="2.0.0",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8080",
    "http://0.0.0.0:8000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
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
    session_id = request_body.session_id
    
    print(f"Received query: {user_query}")
    
    # Create or get session
    if not session_id:
        session_id = session_manager.create_session()
        print(f"Created new session: {session_id}")
    
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Add customer message to history
    session_manager.add_message(session_id, ChatMessage(
        sender="customer",
        content=user_query
    ))
    
    # Check if customer wants to talk to a live agent
    if session_manager.detect_handoff_request(user_query):
        print(f"Handoff request detected for session {session_id}")
        session_manager.request_handoff(session_id)
        
        # Notify all connected agents about new customer in queue
        await connection_manager.broadcast_to_all_agents({
            "type": "new_customer",
            "session_id": session_id,
            "message": "New customer waiting in queue"
        })
        
        response_text = (
            "I'll connect you with a live agent right away. "
            "Please wait a moment while I find someone to help you..."
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
    
    # If session is waiting for agent, remind them
    if session.state == SessionState.WAITING_FOR_AGENT:
        return ChatResponse(
            answer="Please wait, we're connecting you with a live agent...",
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
        
        return ChatResponse(
            answer=agent_answer,
            session_id=session_id,
            state=session.state
        )
    except Exception as e:
        print(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.websocket("/ws/customer/{session_id}")
async def customer_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for customer real-time chat."""
    await connection_manager.connect_customer(session_id, websocket)
    session_manager.set_customer_connected(session_id, True)
    
    try:
        while True:
            # Receive message from customer
            data = await websocket.receive_json()
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
async def agent_login(agent_id: str, name: str):
    """Register/login an agent."""
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
async def get_waiting_queue():
    """Get list of customers waiting for an agent."""
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
async def accept_chat(session_id: str, agent_id: str):
    """Agent accepts a customer chat."""
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
async def end_agent_session_endpoint(session_id: str, return_to_ai: bool = False):
    """End an agent session."""
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
async def get_agent_sessions(agent_id: str):
    """Get all active sessions for an agent."""
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
