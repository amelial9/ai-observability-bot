"""
WebSocket connection manager for real-time chat.
"""
from typing import Dict, List
from fastapi import WebSocket
import json
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections for customers and agents."""
    
    def __init__(self):
        # Customer connections: session_id -> WebSocket
        self.customer_connections: Dict[str, WebSocket] = {}
        
        # Agent connections: agent_id -> WebSocket
        self.agent_connections: Dict[str, WebSocket] = {}
    
    async def connect_customer(self, session_id: str, websocket: WebSocket):
        """Connect a customer WebSocket."""
        await websocket.accept()
        self.customer_connections[session_id] = websocket
        print(f"Customer connected: {session_id}")
    
    async def connect_agent(self, agent_id: str, websocket: WebSocket):
        """Connect an agent WebSocket."""
        await websocket.accept()
        self.agent_connections[agent_id] = websocket
        print(f"Agent connected: {agent_id}")
    
    def disconnect_customer(self, session_id: str):
        """Disconnect a customer."""
        if session_id in self.customer_connections:
            del self.customer_connections[session_id]
            print(f"Customer disconnected: {session_id}")
    
    def disconnect_agent(self, agent_id: str):
        """Disconnect an agent."""
        if agent_id in self.agent_connections:
            del self.agent_connections[agent_id]
            print(f"Agent disconnected: {agent_id}")
    
    async def send_to_customer(self, session_id: str, message: dict):
        """Send message to a customer."""
        if session_id in self.customer_connections:
            try:
                await self.customer_connections[session_id].send_json(message)
            except Exception as e:
                print(f"Error sending to customer {session_id}: {e}")
                self.disconnect_customer(session_id)
    
    async def send_to_agent(self, agent_id: str, message: dict):
        """Send message to an agent."""
        if agent_id in self.agent_connections:
            try:
                await self.agent_connections[agent_id].send_json(message)
            except Exception as e:
                print(f"Error sending to agent {agent_id}: {e}")
                self.disconnect_agent(agent_id)
    
    async def broadcast_to_all_agents(self, message: dict):
        """Broadcast message to all connected agents."""
        disconnected = []
        for agent_id, websocket in self.agent_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to agent {agent_id}: {e}")
                disconnected.append(agent_id)
        
        # Clean up disconnected agents
        for agent_id in disconnected:
            self.disconnect_agent(agent_id)
    
    def is_customer_connected(self, session_id: str) -> bool:
        """Check if customer is connected."""
        return session_id in self.customer_connections
    
    def is_agent_connected(self, agent_id: str) -> bool:
        """Check if agent is connected."""
        return agent_id in self.agent_connections
    
    async def send_typing_indicator(self, session_id: str, is_typing: bool, sender: str):
        """Send typing indicator."""
        message = {
            "type": "typing",
            "is_typing": is_typing,
            "sender": sender,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_to_customer(session_id, message)


# Global connection manager instance
connection_manager = ConnectionManager()
