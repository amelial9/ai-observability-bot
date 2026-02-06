document.addEventListener('DOMContentLoaded', () => {
    let sessionId = null;
    let websocket = null;
    let isConnectedToAgent = false;
    let typingTimeout = null;

    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const chatWidgetButton = document.getElementById('chat-widget-button');
    const chatContainer = document.getElementById('chat-container');
    const closeButton = document.getElementById('close-button');
    const agentStatus = document.getElementById('agent-status');
    const agentNameSpan = document.getElementById('agent-name');
    const typingIndicator = document.getElementById('typing-indicator');

    const API_URL = '/chat';

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');

        if (sender === 'user') {
            messageDiv.classList.add('user-message');
        } else if (sender === 'bot') {
            messageDiv.classList.add('bot-message');
        } else if (sender === 'system') {
            messageDiv.classList.add('system-message');
        }

        messageDiv.textContent = text;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // Typing indicator for customer
    userInput.addEventListener('input', () => {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            // Send typing indicator
            websocket.send(JSON.stringify({
                type: 'typing',
                is_typing: true
            }));

            // Clear previous timeout
            clearTimeout(typingTimeout);

            // Stop typing after 2 seconds of inactivity
            typingTimeout = setTimeout(() => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'typing',
                        is_typing: false
                    }));
                }
            }, 2000);
        }
    });

    async function sendMessage() {
        const query = userInput.value.trim();
        if (query === '') return;

        addMessage(query, 'user');
        userInput.value = '';

        // If connected to live agent via WebSocket, send through WebSocket
        if (isConnectedToAgent && websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
                type: 'message',
                content: query
            }));
            return;
        }

        sendButton.disabled = true;
        loadingIndicator.style.display = 'block';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    session_id: sessionId
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Update session ID
            if (data.session_id) {
                sessionId = data.session_id;
            }

            addMessage(data.answer, 'bot');

            // If state changed to waiting_for_agent, establish WebSocket
            if (data.state === 'waiting_for_agent') {
                connectWebSocket();
            }
        } catch (error) {
            console.error('Error sending message:', error);
            addMessage(`Sorry, there was an error processing your request: ${error.message}`, 'bot');
        } finally {
            sendButton.disabled = false;
            loadingIndicator.style.display = 'none';
            userInput.focus();
        }
    }

    function connectWebSocket() {
        if (!sessionId) {
            console.error('No session ID available');
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/customer/${sessionId}`;

        console.log('Connecting to WebSocket:', wsUrl);
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            console.log('WebSocket connected');
        };

        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('WebSocket message:', data);

            switch (data.type) {
                case 'agent_joined':
                    isConnectedToAgent = true;
                    agentStatus.style.display = 'flex';
                    agentNameSpan.textContent = data.agent_name;
                    addMessage(data.message, 'system');
                    break;

                case 'agent_message':
                    addMessage(data.content, 'bot');
                    break;

                case 'agent_left':
                    isConnectedToAgent = data.return_to_ai || false;
                    if (!isConnectedToAgent) {
                        agentStatus.style.display = 'none';
                    }
                    addMessage(data.message, 'system');
                    break;

                case 'agent_typing':
                    if (data.is_typing) {
                        typingIndicator.style.display = 'flex';
                    } else {
                        typingIndicator.style.display = 'none';
                    }
                    break;

                case 'error':
                    addMessage(data.message, 'system');
                    break;
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        websocket.onclose = () => {
            console.log('WebSocket disconnected');
            // Attempt to reconnect after 3 seconds if still connected to agent
            if (isConnectedToAgent) {
                setTimeout(() => {
                    console.log('Attempting to reconnect...');
                    connectWebSocket();
                }, 3000);
            }
        };
    }

    chatWidgetButton.addEventListener('click', () => {
        chatContainer.classList.remove('chat-container-closed');
        chatContainer.classList.add('chat-container-open');
        userInput.focus();
    });

    closeButton.addEventListener('click', () => {
        chatContainer.classList.remove('chat-container-open');
        chatContainer.classList.add('chat-container-closed');
    });

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });

    console.log('Chat widget initialized');
});