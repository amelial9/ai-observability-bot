"""
Data models for the live agent handoff system.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


class SessionState(str, Enum):
    """Possible states for a chat session."""
    AI = "ai"
    WAITING_FOR_AGENT = "waiting_for_agent"
    LIVE_AGENT = "live_agent"
    ENDED = "ended"


class ChatMessage(BaseModel):
    """Represents a single chat message."""
    sender: Literal["customer", "agent", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    agent_name: Optional[str] = None


class HandoffRequest(BaseModel):
    """Request to handoff a chat to a live agent."""
    session_id: str
    reason: Optional[str] = None


class AgentInfo(BaseModel):
    """Information about a live agent."""
    agent_id: str
    name: str
    is_available: bool = True
    active_sessions: List[str] = Field(default_factory=list)


class SessionInfo(BaseModel):
    """Information about a chat session."""
    session_id: str
    state: SessionState
    customer_connected: bool = False
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    created_at: datetime
    messages: List[ChatMessage] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Request from customer chat endpoint."""
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response to customer chat request."""
    answer: str
    session_id: str
    state: SessionState
    agent_name: Optional[str] = None
