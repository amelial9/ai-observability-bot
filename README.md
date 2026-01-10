# AI Observability Bot

A comprehensive customer engagement solution that combines the **BeeAI Framework** for intelligent FAQ answering, **AI-SIEM** (via OpenLIT) for advanced observability and security monitoring, and **Splunk** integration for enterprise analytics and executive dashboards.

## Overview

This solution addresses the critical need for safe, observable, and scalable AI-powered customer service. It combines three integrated components:

1. **BeeAI Chatbot**: RAG-powered FAQ system that retrieves company knowledge and generates accurate, context-aware responses
2. **AI-SIEM Observability**: Monitors every interaction for hallucinations, jailbreaks, toxicity, bias, prompt injections, and compliance violations
3. **Splunk Integration**: Enterprise-grade analytics platform that transforms telemetry into actionable business insights

Together, these components deliver automated customer service while ensuring visibility, security, and strategic value.

## Key Features

### 🤖 BeeAI Framework Integration
- **RAG-Powered FAQ System**: Combines vector database retrieval with LLM reasoning for accurate, grounded responses
- **Semantic Search**: Finds relevant company knowledge using ChromaDB vector embeddings
- **Context-Aware Responses**: Synthesizes conversational answers from retrieved documents

### 🛡️ AI-SIEM Observability (OpenLIT)
- **Hallucination Detection**: Identifies when the AI generates inaccurate or fabricated information
- **Security Monitoring**: Detects jailbreak attempts, prompt injections, and adversarial inputs
- **Content Safety**: Flags toxic, biased, or inappropriate language
- **Compliance Tracking**: Monitors sensitive topics and regulatory violations
- **Performance Metrics**: Tracks response latency, token usage, and cost per interaction
- **Sidecar Architecture**: Non-intrusive monitoring without middleware dependencies

### 📊 Splunk Analytics Integration
- **OpenTelemetry Export**: Standardized telemetry data sent to Splunk via OTLP
- **Executive Dashboards**: Real-time visibility into chatbot performance, costs, and risks
- **Customer Intelligence**: Aggregates user prompts to reveal trends and unmet needs
- **Alerting**: Automated notifications for security incidents and performance issues
- **Cost Management**: Token usage tracking and optimization insights

### 😊 Customer Experience Features
- **Sentiment Analysis**: Detects customer frustration and triggers escalation
- **Smart Escalation**: Automatically alerts support team after detecting multiple frustrated messages
- **Email Notifications**: Sends detailed conversation summaries to support representatives
- **Chat Widget**: Modern, embeddable chat interface for any website

## Architecture

```
┌─────────────────┐
│   Frontend       │
│  (Chat Widget)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         FastAPI Backend                 │
│  ┌───────────────────────────────────┐  │
│  │      BeeAI Framework              │  │
│  │  • Vector Search (ChromaDB)       │  │
│  │  • LLM Integration (OpenAI)        │  │
│  │  • RAG Orchestration              │  │
│  └──────────────┬────────────────────┘  │
│                 │                        │
│  ┌──────────────▼────────────────────┐  │
│  │   AI-SIEM (OpenLIT SDK)          │  │
│  │  • Evaluations (Hallucinations)   │  │
│  │  • Guardrails (Toxicity, Bias)    │  │
│  │  • Security Detection             │  │
│  │  • Performance Metrics            │  │
│  └──────────────┬────────────────────┘  │
└─────────────────┼────────────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ OpenTelemetry   │
         │   Collector     │
         │  (Docker)       │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Splunk HEC     │
         │    Agent        │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │   Splunk        │
         │  Dashboards     │
         └─────────────────┘
```

## Prerequisites

- Python 3.11 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Gmail account (or other SMTP server) for email notifications
- Excel file with your FAQs (Question and Answer columns)
- **Splunk instance** (optional but recommended for full observability)
- **Docker** (for OpenTelemetry collector container)

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

# Optional: OpenTelemetry Configuration
OTEL_ENDPOINT=http://localhost:4328/v1/traces
SERVICE_NAME=beeai-faq-agent
ENVIRONMENT=production
```

**Note for Gmail users**: You'll need to create an [App Password](https://support.google.com/accounts/answer/185833) instead of using your regular password.

### 4. Prepare Your FAQ Data

1. Create an Excel file (`Pemco_faqs.xlsx`) with two columns:
   - **Question**: The customer's question
   - **Answer**: The answer to that question

2. Place the file in the project root directory

3. Run the extraction script to load FAQs into the vector database:

```bash
python backend/extraction.py
```

This will create a `my_chroma_db` folder with your FAQ embeddings.

### 5. Set Up OpenTelemetry Collector (Optional but Recommended)

For Splunk integration, run the OpenTelemetry collector in Docker:

```bash
# Example Docker command (adjust based on your Splunk setup)
docker run -d \
  -p 4328:4318 \
  -e SPLUNK_HEC_TOKEN=your-hec-token \
  -e SPLUNK_HEC_URL=https://your-splunk-instance:8088 \
  otel/opentelemetry-collector
```

Configure the collector endpoint in `backend/agent.py` if different from `http://localhost:4328`.

### 6. Start the Backend Server

```bash
cd backend
python main.py
```

The server will start on `http://localhost:8001`

**Note**: The system will automatically initialize OpenLIT (AI-SIEM) and begin monitoring all LLM interactions.

### 7. Open the Frontend

Open `frontend/index.html` in your web browser, or serve it through the FastAPI server by visiting `http://localhost:8001/`

## Usage

1. **Click the chat button** in the bottom-right corner of the page
2. **Type your question** in the chat input
3. **Receive AI-powered answers** based on your FAQ database
4. **All interactions are automatically monitored** by AI-SIEM for quality, security, and compliance
5. **Escalation happens automatically** if the system detects frustration (after 3 frustrated messages)
6. **View analytics** in Splunk dashboards for performance, costs, and customer insights

## What AI-SIEM Monitors

Every interaction is analyzed for:

- **Hallucinations**: Detects when the AI generates false or unsupported information
- **Jailbreak Attempts**: Identifies attempts to bypass safety guardrails
- **Prompt Injections**: Detects malicious inputs trying to extract data or manipulate behavior
- **Toxicity**: Flags inappropriate, offensive, or harmful language
- **Bias**: Identifies potentially discriminatory or unfair responses
- **Sensitive Topics**: Monitors for exposure of confidential or regulated information
- **Performance**: Tracks latency, token usage, and cost per interaction
- **Compliance**: Ensures adherence to regulatory requirements

## Splunk Dashboards

Once data flows to Splunk, you can create dashboards showing:

- **Chatbot Performance**: Average response time, success rate, token usage trends
- **Customer Intelligence**: Most common questions, trending topics, unmet needs
- **Security Metrics**: Jailbreak attempts, prompt injections, toxicity incidents over time
- **Cost Analysis**: Token consumption, cost per conversation, optimization opportunities
- **Risk Exposure**: Hallucination frequency, compliance violations, sensitive topic alerts
- **Customer Sentiment**: Frustration levels, escalation patterns, satisfaction trends

## Project Structure

```
ai-observability-bot/
├── backend/
│   ├── main.py              # FastAPI server, sentiment analysis, escalation logic
│   ├── agent.py             # BeeAI Framework RAG system + AI-SIEM (OpenLIT) integration
│   ├── sentiment_analyzer.py # Customer frustration detection
│   ├── email_service.py     # Email notification system
│   └── extraction.py        # Script to load FAQs into ChromaDB
├── frontend/
│   ├── index.html           # Main HTML page
│   └── static/
│       ├── script.js        # Chat widget JavaScript
│       └── style.css        # Chat widget styling
├── my_chroma_db/            # ChromaDB vector database (created after extraction)
├── Pemco_faqs.xlsx          # Your FAQ data (you provide this)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
└── README.md               # This file
```

## API Endpoints

- `GET /` - Serves the frontend HTML page
- `POST /chat` - Main chat endpoint (receives user messages, returns AI responses with sentiment analysis)
- `GET /session/{session_id}/history` - Get conversation history for a session
- `POST /session/reset` - Reset the conversation session
- `GET /health` - Health check endpoint (shows system status)

## Configuration Options

### Adjusting Frustration Detection

In `backend/main.py`:

```python
# How sensitive is frustration detection? (0.0 = calm, 1.0 = very frustrated)
sentiment_analyzer = SentimentAnalyzer(frustration_threshold=0.6)

# How many frustrated messages trigger escalation?
conversation_tracker = ConversationTracker(trigger_count=3)
```

### Changing the AI Model

In `backend/agent.py`:

```python
# Change from gpt-4o to gpt-3.5-turbo to save costs
_llm = ChatModel.from_name("openai:gpt-3.5-turbo")
```

### Configuring AI-SIEM (OpenLIT)

In `backend/agent.py`, the OpenLIT initialization includes:

```python
# OpenTelemetry endpoint (where Splunk collector is listening)
OTEL_ENDPOINT = "http://localhost:4328"

# Service name (appears in Splunk)
SERVICE_NAME = "beeai-faq-agent"

# Environment tag
ENVIRONMENT = "production"
```

### Customizing the Chat Widget

Edit `frontend/static/style.css` to change colors, sizes, and positioning.

## Docker Deployment

### Build the Application

```bash
docker build -t ai-chatbot .
```

### Run the Container

```bash
docker run -p 8001:8001 --env-file .env ai-chatbot
```

### Run OpenTelemetry Collector

```bash
docker run -d \
  -p 4328:4318 \
  -e SPLUNK_HEC_TOKEN=your-token \
  -e SPLUNK_HEC_URL=https://your-splunk:8088 \
  otel/opentelemetry-collector
```

## Troubleshooting

### "OpenAI API key not found"
- Make sure you've created a `.env` file with `OPENAI_API_KEY`
- Verify the key is correct and has credits

### "OpenLIT initialization failed"
- Check that `OTEL_ENDPOINT` is correct and the OpenTelemetry collector is running
- Verify network connectivity to the collector endpoint
- Check Docker container logs if using Docker

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

### Splunk not receiving data
- Verify the OpenTelemetry collector is running and accessible
- Check that `OTEL_ENDPOINT` in `agent.py` matches your collector address
- Verify Splunk HEC token and URL are correct in the collector configuration
- Check collector logs for connection errors

## Observability & Monitoring

### OpenTelemetry Integration

The system automatically tracks:
- **Session IDs**: Unique identifiers for each conversation
- **Sentiment Scores**: Customer frustration levels
- **Escalation Events**: When human support is triggered
- **RAG Tool Execution**: Vector search performance and results
- **LLM Interactions**: Prompts, responses, token usage, latency
- **AI-SIEM Evaluations**: Hallucination scores, guardrail violations
- **Security Events**: Jailbreak attempts, prompt injections
- **Email Notifications**: Escalation email send status

### Data Flow

1. **User Interaction** → Captured by FastAPI
2. **BeeAI Processing** → RAG search and LLM generation
3. **AI-SIEM Analysis** → OpenLIT evaluates and monitors
4. **OpenTelemetry Export** → Standardized telemetry sent to collector
5. **Splunk Ingestion** → HEC agent forwards to Splunk
6. **Dashboards & Alerts** → Visualizations and notifications

## Business Benefits

- **Cost Reduction**: Automate routine customer inquiries, reducing support team workload
- **Risk Mitigation**: Continuous monitoring prevents hallucinations, toxicity, and security breaches
- **Customer Intelligence**: Analyze prompts to identify trends, unmet needs, and product opportunities
- **Compliance Assurance**: Track and prevent regulatory violations automatically
- **Performance Optimization**: Monitor token usage and latency to optimize costs and speed
- **Strategic Insights**: Executive dashboards enable data-driven decision making

## Cost Considerations

- **OpenAI API**: Approximately $0.01-$0.10 per conversation (depends on model and message length)
- **Email**: Free with Gmail
- **Hosting**: Varies by provider (can run locally for free)
- **Splunk**: Depends on your Splunk license and data volume

**Cost Optimization Tips**:
- Use `gpt-3.5-turbo` instead of `gpt-4o` for lower costs
- Monitor token usage in Splunk dashboards
- Optimize FAQ database to reduce LLM dependency
- Use caching for common queries

## Future Directions

This solution provides a foundation for evolving toward:

- **Agentic AI Systems**: Multi-agent workflows with autonomous decision-making
- **LLM Assembly Lines**: Structured pipelines for complex reasoning tasks
- **Advanced Orchestration**: Collaborative agents working together
- **Continuous Learning**: Retraining models based on collected interaction data

AI-SIEM ensures that as these systems evolve, they remain safe, observable, and compliant.

## License

[Add your license here]

## Support

For issues or questions, please [open an issue](link-to-issues) or contact [your contact info].

## References

- [BeeAI Framework Documentation](link-to-beeai-docs)
- [OpenLIT Documentation](https://docs.openlit.io)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Splunk HEC Documentation](https://docs.splunk.com/Documentation/Splunk/latest/Data/HECExamples)
