# AI Observability Bot

An intelligent customer support chatbot that answers FAQs using AI, detects customer frustration, and automatically escalates to human support when needed.

## Features

- 🤖 **RAG-Powered FAQ System**: Answers customer questions using your company's FAQ database
- 😊 **Sentiment Analysis**: Automatically detects when customers are frustrated
- 🚨 **Smart Escalation**: Alerts support team after detecting multiple frustrated messages
- 📧 **Email Notifications**: Sends detailed conversation summaries to support representatives
- 📊 **Observability**: Tracks all interactions with OpenTelemetry for monitoring and analytics
- 💬 **Chat Widget**: Clean, modern chat interface that can be embedded on any website

## Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Gmail account (or other SMTP server) for email notifications
- Excel file with your FAQs (Question and Answer columns)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-observability-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Required: Email Configuration (for escalation notifications)
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-gmail-app-password
RECIPIENT_EMAIL=support@yourcompany.com

# Optional: SMTP Settings (defaults to Gmail)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

**Note for Gmail users**: You'll need to create an [App Password](https://support.google.com/accounts/answer/185833) instead of using your regular password.

### 4. Prepare Your FAQ Data

1. Create an Excel file (`Pemco_faqs.xlsx`) with two columns:
   - **Question**: The customer's question
   - **Answer**: The answer to that question

2. Place the file in the project root directory

3. Run the extraction script to load FAQs into the database:

```bash
python backend/extraction.py
```

This will create a `my_chroma_db` folder with your FAQ embeddings.

### 5. Start the Backend Server

```bash
cd backend
python main.py
```

The server will start on `http://localhost:8001`

### 6. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it through the FastAPI server by visiting `http://localhost:8001/`

## Usage

1. **Click the chat button** in the bottom-right corner of the page
2. **Type your question** in the chat input
3. **Receive AI-powered answers** based on your FAQ database
4. **Escalation happens automatically** if the system detects frustration (after 3 frustrated messages)

## Project Structure

```
ai-observability-bot/
├── backend/
│   ├── main.py              # FastAPI server and API endpoints
│   ├── agent.py             # RAG system for FAQ answering
│   ├── sentiment_analyzer.py # Frustration detection
│   ├── email_service.py     # Email notification system
│   └── extraction.py        # Script to load FAQs into database
├── frontend/
│   ├── index.html           # Main HTML page
│   └── static/
│       ├── script.js        # Chat widget JavaScript
│       └── style.css        # Chat widget styling
├── my_chroma_db/            # Vector database (created after running extraction.py)
├── Pemco_faqs.xlsx          # Your FAQ data (you provide this)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
└── README.md               # This file
```

## API Endpoints

- `GET /` - Serves the frontend HTML page
- `POST /chat` - Main chat endpoint (receives user messages, returns AI responses)
- `GET /session/{session_id}/history` - Get conversation history for a session
- `POST /session/reset` - Reset the conversation session
- `GET /health` - Health check endpoint

## Configuration Options

### Adjusting Frustration Detection

In `backend/main.py`, you can customize:

```python
# How sensitive is frustration detection? (0.0 = calm, 1.0 = very frustrated)
sentiment_analyzer = SentimentAnalyzer(frustration_threshold=0.6)

# How many frustrated messages trigger escalation?
conversation_tracker = ConversationTracker(trigger_count=3)
```

### Changing the AI Model

In `backend/agent.py`, you can switch to a cheaper model:

```python
# Change from gpt-4o to gpt-3.5-turbo to save costs
_llm = ChatModel.from_name("openai:gpt-3.5-turbo")
```

### Customizing the Chat Widget

Edit `frontend/static/style.css` to change colors, sizes, and positioning.

## Docker Deployment

Build the Docker image:

```bash
docker build -t ai-chatbot .
```

Run the container:

```bash
docker run -p 8001:8001 --env-file .env ai-chatbot
```

## Troubleshooting

### "OpenAI API key not found"
- Make sure you've created a `.env` file with `OPENAI_API_KEY`
- Verify the key is correct and has credits

### "Email not sending"
- For Gmail, use an App Password (not your regular password)
- Check that `SENDER_EMAIL`, `SENDER_PASSWORD`, and `RECIPIENT_EMAIL` are set correctly
- Verify SMTP server and port settings

### "No FAQs found" or empty responses
- Run `python backend/extraction.py` to populate the database
- Check that your Excel file has "Question" and "Answer" columns
- Verify the Excel file path is correct

### "Frontend can't connect to backend"
- Make sure the backend server is running on port 8001
- Check `API_URL` in `frontend/static/script.js` matches your backend address
- Verify CORS settings in `backend/main.py` if hosting on different domains

### ChromaDB errors
- Ensure the `my_chroma_db` folder exists and is writable
- Try deleting the folder and re-running `extraction.py`

## Observability

The system uses OpenTelemetry to track:
- Session IDs
- Sentiment scores
- Frustration counts
- Escalation events
- Email notifications
- RAG tool execution

Traces are sent to `http://localhost:4328` by default. Configure your OpenTelemetry endpoint in `backend/agent.py` if needed.

## Cost Considerations

- **OpenAI API**: Approximately $0.01-$0.10 per conversation (depends on model and message length)
- **Email**: Free with Gmail
- **Hosting**: Varies by provider (can run locally for free)

To reduce costs, consider using `gpt-3.5-turbo` instead of `gpt-4o` in `agent.py`.

## License

[Add your license here]

## Support

For issues or questions, please [open an issue](link-to-issues) or contact [your contact info].
