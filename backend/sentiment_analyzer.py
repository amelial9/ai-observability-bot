# backend/sentiment_analyzer.py
"""
Sentiment Analysis Module for Customer Frustration Detection
Analyzes user messages to detect frustration levels and trigger escalation when needed.
"""

import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import enhanced observability module
try:
    from observability import (
        trace_operation,
        record_sentiment_score,
        record_llm_tokens,
        add_span_attributes
    )
    ENHANCED_OBSERVABILITY = True
    print("✅ Sentiment analyzer: Enhanced observability loaded")
except ImportError:
    ENHANCED_OBSERVABILITY = False
    print("ℹ️  Sentiment analyzer: Running without enhanced observability")

class SentimentAnalyzer:
    """
    Analyzes sentiment of user messages to detect frustration levels.
    Uses OpenAI to score messages on a 0-1 scale where:
    - 0.0-0.3: Neutral/Positive
    - 0.3-0.6: Slightly frustrated
    - 0.6-0.8: Moderately frustrated
    - 0.8-1.0: Highly frustrated
    """

    def __init__(self, frustration_threshold: float = 0.6):
        """
        Initialize the sentiment analyzer.

        Args:
            frustration_threshold: Score above which a message is considered frustrated (default: 0.6)
        """
        self.frustration_threshold = frustration_threshold

    async def analyze_sentiment(self, user_message: str) -> float:
        """
        Analyze the sentiment of a user message and return frustration score.

        Args:
            user_message: The user's input text to analyze

        Returns:
            float: Frustration score between 0.0 and 1.0
        """
        if ENHANCED_OBSERVABILITY:
            with trace_operation(
                "sentiment_analysis",
                {
                    "sentiment.message_length": len(user_message),
                    "sentiment.message_preview": user_message[:100]
                }
            ) as span:
                try:
                    start_time = time.time()

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",  # Using mini for cost efficiency
                        messages=[
                            {
                                "role": "system",
                                "content": """You are a sentiment analysis expert specializing in detecting customer frustration.
Analyze the user's message and rate their frustration level on a scale from 0.0 to 1.0:

0.0-0.3: Neutral, calm, polite, or positive tone
0.3-0.6: Slightly frustrated, impatient, or mildly annoyed
0.6-0.8: Moderately frustrated, angry, or expressing significant dissatisfaction
0.8-1.0: Highly frustrated, very angry, using aggressive language, or expressing severe dissatisfaction

Consider these indicators of frustration:
- Negative words (terrible, awful, worst, hate, frustrated, angry)
- Caps lock or excessive punctuation (!!!, ???)
- Demanding language or urgency
- Complaints about repeated issues
- Expressions of wasted time
- Threats to leave or escalate
- Sarcasm or passive aggression

Respond with ONLY a number between 0.0 and 1.0, nothing else."""
                            },
                            {
                                "role": "user",
                                "content": f"Rate the frustration level of this message: \"{user_message}\""
                            }
                        ],
                        temperature=0.3,  # Low temperature for consistent scoring
                        max_tokens=10
                    )

                    analysis_time_ms = (time.time() - start_time) * 1000

                    score_text = response.choices[0].message.content.strip()

                    # Parse the score
                    try:
                        score = float(score_text)
                        # Ensure score is within valid range
                        score = max(0.0, min(1.0, score))

                        # Determine category
                        if score < 0.3:
                            category = "neutral_positive"
                        elif score < 0.6:
                            category = "slightly_frustrated"
                        elif score < 0.8:
                            category = "moderately_frustrated"
                        else:
                            category = "highly_frustrated"

                        # Record sentiment metrics
                        record_sentiment_score(score, category)

                        # Record LLM token usage (estimated)
                        estimated_prompt_tokens = 200  # System prompt is ~200 tokens
                        estimated_completion_tokens = 2  # Response is just a number
                        record_llm_tokens(
                            prompt_tokens=estimated_prompt_tokens,
                            completion_tokens=estimated_completion_tokens,
                            model="gpt-4o-mini"
                        )

                        if span:
                            span.set_attribute("sentiment.score", score)
                            span.set_attribute("sentiment.category", category)
                            span.set_attribute("sentiment.analysis_time_ms", analysis_time_ms)
                            span.set_attribute("sentiment.model", "gpt-4o-mini")
                            span.set_attribute("sentiment.is_frustrated", score >= self.frustration_threshold)

                        return score
                    except ValueError:
                        print(f"Warning: Could not parse sentiment score '{score_text}', defaulting to 0.0")
                        if span:
                            span.set_attribute("sentiment.parse_error", True)
                        return 0.0

                except Exception as e:
                    print(f"Error analyzing sentiment: {e}")
                    if span:
                        span.set_attribute("sentiment.error", str(e))
                    return 0.0  # Default to neutral if analysis fails
        else:
            # Fallback without enhanced observability
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a sentiment analysis expert specializing in detecting customer frustration.
Analyze the user's message and rate their frustration level on a scale from 0.0 to 1.0:

0.0-0.3: Neutral, calm, polite, or positive tone
0.3-0.6: Slightly frustrated, impatient, or mildly annoyed
0.6-0.8: Moderately frustrated, angry, or expressing significant dissatisfaction
0.8-1.0: Highly frustrated, very angry, using aggressive language, or expressing severe dissatisfaction

Consider these indicators of frustration:
- Negative words (terrible, awful, worst, hate, frustrated, angry)
- Caps lock or excessive punctuation (!!!, ???)
- Demanding language or urgency
- Complaints about repeated issues
- Expressions of wasted time
- Threats to leave or escalate
- Sarcasm or passive aggression

Respond with ONLY a number between 0.0 and 1.0, nothing else."""
                        },
                        {
                            "role": "user",
                            "content": f"Rate the frustration level of this message: \"{user_message}\""
                        }
                    ],
                    temperature=0.3,
                    max_tokens=10
                )

                score_text = response.choices[0].message.content.strip()

                try:
                    score = float(score_text)
                    score = max(0.0, min(1.0, score))
                    return score
                except ValueError:
                    print(f"Warning: Could not parse sentiment score '{score_text}', defaulting to 0.0")
                    return 0.0

            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
                return 0.0


class ConversationTracker:
    """
    Tracks conversation history and sentiment scores across multiple messages.
    Detects when escalation is needed based on frustration patterns.
    """

    def __init__(self, frustration_threshold: float = 0.6, trigger_count: int = 3):
        """
        Initialize the conversation tracker.

        Args:
            frustration_threshold: Score above which a message is considered frustrated
            trigger_count: Number of frustrated messages needed to trigger escalation
        """
        self.frustration_threshold = frustration_threshold
        self.trigger_count = trigger_count

        # Storage for conversation data (in production, use database)
        self.conversations: Dict[str, Dict[str, Any]] = {}

    def track_message(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        sentiment_score: float
    ) -> Dict[str, Any]:
        """
        Track a message and its sentiment in the conversation history.

        Args:
            session_id: Unique identifier for the conversation session
            user_message: The user's input message
            bot_response: The bot's response
            sentiment_score: The calculated frustration score

        Returns:
            Dict containing tracking info and escalation status
        """
        # Initialize conversation if it doesn't exist
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                "messages": [],
                "sentiment_scores": [],
                "frustrated_count": 0,
                "escalated": False,
                "start_time": datetime.now().isoformat()
            }

        conversation = self.conversations[session_id]

        # Add message to history
        conversation["messages"].append({
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "sentiment_score": sentiment_score
        })

        conversation["sentiment_scores"].append(sentiment_score)

        # Check if this message indicates frustration
        if sentiment_score >= self.frustration_threshold:
            conversation["frustrated_count"] += 1

        # Check if escalation should be triggered
        should_escalate = (
            not conversation["escalated"] and
            conversation["frustrated_count"] >= self.trigger_count
        )

        if should_escalate:
            conversation["escalated"] = True

        return {
            "session_id": session_id,
            "current_score": sentiment_score,
            "frustrated_count": conversation["frustrated_count"],
            "should_escalate": should_escalate,
            "total_messages": len(conversation["messages"])
        }

    def get_conversation_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the full conversation history for a session.

        Args:
            session_id: The conversation session ID

        Returns:
            Dictionary containing full conversation data or None if not found
        """
        return self.conversations.get(session_id)

    def reset_conversation(self, session_id: str):
        """
        Reset/clear a conversation session.

        Args:
            session_id: The conversation session ID to reset
        """
        if session_id in self.conversations:
            del self.conversations[session_id]


async def generate_conversation_summary(conversation_history: Dict[str, Any]) -> str:
    """
    Generate a summary of the conversation for the representative.

    Args:
        conversation_history: Full conversation data including messages and scores

    Returns:
        str: Formatted summary of the conversation
    """
    try:
        messages = conversation_history.get("messages", [])

        # Build conversation text
        conversation_text = []
        for msg in messages:
            conversation_text.append(f"User: {msg['user_message']}")
            conversation_text.append(f"Bot: {msg['bot_response']}")
            conversation_text.append(f"Frustration Score: {msg['sentiment_score']:.2f}")
            conversation_text.append("")

        full_conversation = "\n".join(conversation_text)

        # Use OpenAI to generate summary
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a customer service manager reviewing escalated conversations.
Provide a concise summary that includes:
1. Main issue or question the customer has
2. Key frustration points or problems encountered
3. Current status of the issue
4. Recommended next steps for the representative

Keep the summary professional, factual, and under 200 words."""
                },
                {
                    "role": "user",
                    "content": f"Summarize this customer conversation:\n\n{full_conversation}"
                }
            ],
            temperature=0.5,
            max_tokens=300
        )

        summary = response.choices[0].message.content.strip()

        # Add metadata
        avg_score = sum(conversation_history["sentiment_scores"]) / len(conversation_history["sentiment_scores"])
        max_score = max(conversation_history["sentiment_scores"])

        full_summary = f"""ESCALATED CONVERSATION SUMMARY
{'=' * 50}

{summary}

FRUSTRATION METRICS:
- Average Frustration Score: {avg_score:.2f}
- Peak Frustration Score: {max_score:.2f}
- Messages with High Frustration: {conversation_history['frustrated_count']}
- Total Messages: {len(messages)}

CONVERSATION START TIME: {conversation_history['start_time']}
"""

        return full_summary

    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Error generating summary: {e}"