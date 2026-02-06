// Agent Dashboard Logic

let agentId = null;
let agentName = null;
let websocket = null;
let activeSessionId = null;
let queueCheckInterval = null;
let typingTimeout = null;

const loginScreen = document.getElementById('login-screen');
const dashboard = document.getElementById('dashboard');
const agentNameInput = document.getElementById('agent-name-input');
const loginButton = document.getElementById('login-button');
const loggedAgentName = document.getElementById('logged-agent-name');
const logoutButton = document.getElementById('logout-button');
const queueList = document.getElementById('queue-list');
const queueCount = document.getElementById('queue-count');
const activeChat = document.getElementById('active-chat');
const noChatSelected = document.getElementById('no-chat-selected');
const customerSessionId = document.getElementById('customer-session-id');
const chatMessages = document.getElementById('chat-messages');
const agentMessageInput = document.getElementById('agent-message-input');
const sendMessageButton = document.getElementById('send-message-button');
const endChatButton = document.getElementById('end-chat-button');
const returnToAiButton = document.getElementById('return-to-ai-button');
const customerTypingIndicator = document.getElementById('customer-typing-indicator');

// Login
loginButton.addEventListener('click', login);
agentNameInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') login();
});

async function login() {
    const name = agentNameInput.value.trim();
    if (!name) return;

    agentName = name;
    agentId = 'agent_' + Math.random().toString(36).substr(2, 9);

    try {
        const response = await fetch(`/api/agent/login?agent_id=${agentId}&name=${encodeURIComponent(name)}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.success) {
            showDashboard();
            connectWebSocket();
            startQueuePolling();
        }
    } catch (error) {
        console.error('Login error:', error);
        alert('Failed to login');
    }
}

function showDashboard() {
    loginScreen.style.display = 'none';
    dashboard.style.display = 'block';
    loggedAgentName.textContent = agentName;
}

// WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/agent/${agentId}`;

    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
        console.log('Agent WebSocket connected');
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Agent WebSocket message:', data);

        if (data.type === 'new_customer') {
            fetchQueue();
        } else if (data.session_id === activeSessionId) {
            if (data.type === 'customer_message') {
                addMessage(data.content, 'customer');
            } else if (data.type === 'customer_typing') {
                if (data.is_typing) {
                    customerTypingIndicator.style.display = 'block';
                } else {
                    customerTypingIndicator.style.display = 'none';
                }
            }
        }
    };

    websocket.onclose = () => {
        console.log('Agent WebSocket disconnected');
        setTimeout(connectWebSocket, 3000);
    };
}

// Queue Polling
function startQueuePolling() {
    fetchQueue();
    queueCheckInterval = setInterval(fetchQueue, 5000);
}

async function fetchQueue() {
    try {
        const response = await fetch('/api/agent/queue');
        const data = await response.json();
        renderQueue(data.queue);
    } catch (error) {
        console.error('Error fetching queue:', error);
    }
}

function renderQueue(queue) {
    queueCount.textContent = queue.length;

    if (queue.length === 0) {
        queueList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-inbox"></i>
                <p>No customers waiting</p>
            </div>
        `;
        return;
    }

    queueList.innerHTML = '';
    queue.forEach(item => {
        const queueItem = document.createElement('div');
        queueItem.classList.add('queue-item');
        queueItem.innerHTML = `
            <div class="queue-item-header">
                <span class="session-label">Session: ${item.session_id.substr(0, 8)}...</span>
                <span class="msg-count">${item.message_count} msgs</span>
            </div>
            <div class="last-msg">${item.last_message || 'No messages yet'}</div>
            <button class="btn-accept" onclick="acceptChat('${item.session_id}')">Accept Chat</button>
        `;
        queueList.appendChild(queueItem);
    });
}

// Chat Actions
window.acceptChat = async (sessionId) => {
    try {
        const response = await fetch(`/api/agent/accept/${sessionId}?agent_id=${agentId}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.success) {
            activeSessionId = sessionId;
            showChat(data.session);
            fetchQueue();
        }
    } catch (error) {
        console.error('Error accepting chat:', error);
    }
};

function showChat(session) {
    noChatSelected.style.display = 'none';
    activeChat.style.display = 'flex';
    customerSessionId.textContent = `Customer: ${session.session_id.substr(0, 8)}...`;

    chatMessages.innerHTML = '';
    session.messages.forEach(msg => {
        addMessage(msg.content, msg.sender);
    });
}

sendMessageButton.addEventListener('click', sendMessage);
agentMessageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Typing indicator for agent
agentMessageInput.addEventListener('input', () => {
    if (websocket && websocket.readyState === WebSocket.OPEN && activeSessionId) {
        websocket.send(JSON.stringify({
            type: 'typing',
            session_id: activeSessionId,
            is_typing: true
        }));

        clearTimeout(typingTimeout);
        typingTimeout = setTimeout(() => {
            if (websocket && websocket.readyState === WebSocket.OPEN && activeSessionId) {
                websocket.send(JSON.stringify({
                    type: 'typing',
                    session_id: activeSessionId,
                    is_typing: false
                }));
            }
        }, 2000);
    }
});

function sendMessage() {
    const content = agentMessageInput.value.trim();
    if (!content || !activeSessionId) return;

    websocket.send(JSON.stringify({
        type: 'message',
        session_id: activeSessionId,
        content: content
    }));

    addMessage(content, 'agent');
    agentMessageInput.value = '';
}

function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message');
    messageDiv.classList.add(sender === 'agent' ? 'agent-msg' : (sender === 'system' ? 'system-msg' : 'customer-msg'));
    messageDiv.textContent = content;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

endChatButton.addEventListener('click', () => endChat(false));
returnToAiButton.addEventListener('click', () => endChat(true));

async function endChat(returnToAi) {
    if (!activeSessionId) return;

    try {
        const response = await fetch(`/api/agent/end/${activeSessionId}?return_to_ai=${returnToAi}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (data.success) {
            activeSessionId = null;
            activeChat.style.display = 'none';
            noChatSelected.style.display = 'flex';
        }
    } catch (error) {
        console.error('Error ending chat:', error);
    }
}

logoutButton.addEventListener('click', () => location.reload());
