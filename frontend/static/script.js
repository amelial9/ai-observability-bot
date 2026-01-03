// frontend/static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const loadingIndicator = document.getElementById('loading-indicator');
    const chatWidgetButton = document.getElementById('chat-widget-button');
    const chatContainer = document.getElementById('chat-container');
    const closeButton = document.getElementById('close-button');
    
    const API_URL = 'http://127.0.0.1:8001/chat';
    
    // Generate session ID once per page load
    let sessionId = sessionStorage.getItem('chat_session_id');
    if (!sessionId) {
        sessionId = generateSessionId();
        sessionStorage.setItem('chat_session_id', sessionId);
    }
    
    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.textContent = text;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    
    function addEscalationNotice(escalationMessage) {
        const noticeDiv = document.createElement('div');
        noticeDiv.classList.add('message', 'system-message');
        noticeDiv.style.cssText = `
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 16px;
            margin: 12px 0;
            border-radius: 12px;
            border-left: 4px solid #f59e0b;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;
        
        const icon = document.createElement('div');
        icon.style.cssText = 'font-size: 24px; margin-bottom: 8px;';
        icon.textContent = '🔔';
        
        const title = document.createElement('div');
        title.style.cssText = 'font-weight: bold; font-size: 14px; margin-bottom: 8px;';
        title.textContent = 'Support Team Notified';
        
        const message = document.createElement('div');
        message.style.cssText = 'font-size: 13px; line-height: 1.5;';
        message.textContent = escalationMessage;
        
        noticeDiv.appendChild(icon);
        noticeDiv.appendChild(title);
        noticeDiv.appendChild(message);
        
        chatHistory.appendChild(noticeDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    
    async function sendMessage() {
        const query = userInput.value.trim();
        if (query === '') return;
        
        // Display user message immediately
        addMessage(query, 'user');
        userInput.value = '';
        
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
            
            // Display bot response
            addMessage(data.answer, 'bot');
            
            // Display escalation notice if triggered
            if (data.should_escalate && data.escalation_message) {
                addEscalationNotice(data.escalation_message);
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
    
    // Add initial welcome message
    setTimeout(() => {
        addMessage('Hello! How can I help you today? Feel free to ask me any questions about our company FAQs.', 'bot');
    }, 500);
});