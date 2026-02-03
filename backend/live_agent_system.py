"""
Live agent handoff system with session management.
"""
import uuid
from datetime import datetime
from typing import Dict, Optional, List
from models import SessionState, SessionInfo, ChatMessage, AgentInfo


# Keywords that trigger handoff to live agent
HANDOFF_KEYWORDS = [
    "representative",
    "live agent",
    "talk to someone",
    "human",
    "speak to agent",
    "customer service",
    "real person",
    "talk to a person",
    "speak to a human",
    "customer support",
]


class SessionManager:
    """Manages chat sessions and agent assignments."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionInfo] = {}
        self.agents: Dict[str, AgentInfo] = {}
        self.waiting_queue: List[str] = []  # Session IDs waiting for agent
    
    def create_session(self) -> str:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        session = SessionInfo(
            session_id=session_id,
            state=SessionState.AI,
            created_at=datetime.now(),
            messages=[]
        )
        self.sessions[session_id] = session
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, message: ChatMessage):
        """Add a message to session history."""
        session = self.get_session(session_id)
        if session:
            session.messages.append(message)
    
    def detect_handoff_request(self, text: str) -> bool:
        """Check if text contains handoff keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in HANDOFF_KEYWORDS)
    
    def request_handoff(self, session_id: str) -> bool:
        """Request handoff to live agent."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.state = SessionState.WAITING_FOR_AGENT
        if session_id not in self.waiting_queue:
            self.waiting_queue.append(session_id)
        return True
    
    def register_agent(self, agent_id: str, name: str) -> AgentInfo:
        """Register a new agent."""
        agent = AgentInfo(
            agent_id=agent_id,
            name=name,
            is_available=True,
            active_sessions=[]
        )
        self.agents[agent_id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent by ID."""
        return self.agents.get(agent_id)
    
    def assign_agent(self, session_id: str, agent_id: str) -> bool:
        """Assign an agent to a session."""
        session = self.get_session(session_id)
        agent = self.get_agent(agent_id)
        
        if not session or not agent:
            return False
        
        session.state = SessionState.LIVE_AGENT
        session.agent_id = agent_id
        session.agent_name = agent.name
        
        agent.active_sessions.append(session_id)
        
        # Remove from waiting queue
        if session_id in self.waiting_queue:
            self.waiting_queue.remove(session_id)
        
        # Add system message
        self.add_message(session_id, ChatMessage(
            sender="system",
            content=f"Connected to {agent.name}",
            agent_name=agent.name
        ))
        
        return True
    
    def end_agent_session(self, session_id: str, return_to_ai: bool = False) -> bool:
        """End agent session."""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Remove from agent's active sessions
        if session.agent_id:
            agent = self.get_agent(session.agent_id)
            if agent and session_id in agent.active_sessions:
                agent.active_sessions.remove(session_id)
        
        # Update session state
        if return_to_ai:
            session.state = SessionState.AI
            session.agent_id = None
            session.agent_name = None
            self.add_message(session_id, ChatMessage(
                sender="system",
                content="Agent has left the chat. You can continue with the AI assistant."
            ))
        else:
            session.state = SessionState.ENDED
            self.add_message(session_id, ChatMessage(
                sender="system",
                content="Chat session ended. Thank you!"
            ))
        
        return True
    
    def get_waiting_sessions(self) -> List[SessionInfo]:
        """Get all sessions waiting for an agent."""
        return [
            self.sessions[sid] 
            for sid in self.waiting_queue 
            if sid in self.sessions
        ]
    
    def get_agent_sessions(self, agent_id: str) -> List[SessionInfo]:
        """Get all active sessions for an agent."""
        agent = self.get_agent(agent_id)
        if not agent:
            return []
        
        return [
            self.sessions[sid]
            for sid in agent.active_sessions
            if sid in self.sessions
        ]
    
    def set_customer_connected(self, session_id: str, connected: bool):
        """Update customer connection status."""
        session = self.get_session(session_id)
        if session:
            session.customer_connected = connected
    
    def cleanup_disconnected_sessions(self):
        """Clean up sessions where customer disconnected."""
        disconnected = [
            sid for sid, session in self.sessions.items()
            if not session.customer_connected and session.state != SessionState.ENDED
        ]
        
        for session_id in disconnected:
            # If in live agent mode, notify and end
            session = self.sessions[session_id]
            if session.state == SessionState.LIVE_AGENT:
                self.end_agent_session(session_id, return_to_ai=False)


# Global session manager instance
session_manager = SessionManager()
