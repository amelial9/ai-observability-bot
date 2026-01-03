# backend/email_service.py
"""
Email Service for Sending Escalation Notifications
Sends chat summaries to support representatives via email.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime

class EmailService:
    """
    Service for sending escalation emails to support representatives.
    Supports Gmail SMTP by default, easily adaptable to other providers.
    """
    
    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        recipient_email: Optional[str] = None
    ):
        """
        Initialize the email service.
        
        Args:
            smtp_server: SMTP server address (defaults to Gmail)
            smtp_port: SMTP port (defaults to 587 for TLS)
            sender_email: Email address to send from
            sender_password: Password or app password for sender email
            recipient_email: Default recipient email for escalations
        """
        # Use environment variables or provided values
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = sender_email or os.getenv("SENDER_EMAIL")
        self.sender_password = sender_password or os.getenv("SENDER_PASSWORD")
        self.recipient_email = recipient_email or os.getenv("RECIPIENT_EMAIL")
        
        # Validate configuration
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            print("Warning: Email service not fully configured. Check environment variables:")
            print("  - SENDER_EMAIL")
            print("  - SENDER_PASSWORD") 
            print("  - RECIPIENT_EMAIL")
    
    def send_escalation_email(
        self,
        conversation_summary: str,
        session_id: str,
        recipient_email: Optional[str] = None
    ) -> bool:
        """
        Send an escalation email with conversation summary.
        
        Args:
            conversation_summary: The formatted conversation summary text
            session_id: The conversation session ID for reference
            recipient_email: Override recipient (uses default if not provided)
            
        Returns:
            bool: True if email sent successfully, False otherwise
        """
        # Check configuration
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            print("Error: Email service not configured. Cannot send email.")
            print(f"Would have sent this summary:\n{conversation_summary}")
            return False
        
        # Use provided recipient or default
        to_email = recipient_email or self.recipient_email
        
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"⚠️ Customer Escalation Alert - Session {session_id[:8]}"
            message["From"] = self.sender_email
            message["To"] = to_email
            message["Date"] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
            
            # Create plain text version
            text_content = f"""
Customer Escalation Alert
========================

A customer conversation has been escalated due to detected frustration.

Session ID: {session_id}
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{conversation_summary}

Please review this conversation and reach out to the customer as soon as possible.

---
This is an automated message from the Customer Support Chatbot System.
"""
            
            # Create HTML version for better formatting
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background-color: #ff6b6b;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .metadata {{
            background-color: #f7f7f7;
            padding: 15px;
            border-left: 4px solid #ff6b6b;
            margin-bottom: 20px;
        }}
        .summary {{
            white-space: pre-wrap;
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 12px;
            color: #666;
        }}
        .alert-badge {{
            background-color: #fff;
            color: #ff6b6b;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
            display: inline-block;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚠️ Customer Escalation Alert</h1>
        <p style="margin: 10px 0 0 0;">Immediate attention required</p>
    </div>
    
    <div class="metadata">
        <p><strong>Session ID:</strong> {session_id}</p>
        <p><strong>Escalation Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><span class="alert-badge">HIGH PRIORITY</span></p>
    </div>
    
    <div class="summary">
{conversation_summary}
    </div>
    
    <div class="footer">
        <p><strong>Next Steps:</strong></p>
        <ul>
            <li>Review the conversation summary above</li>
            <li>Reach out to the customer within 1 hour</li>
            <li>Document the resolution in the CRM system</li>
        </ul>
        <p><em>This is an automated message from the Customer Support Chatbot System.</em></p>
    </div>
</body>
</html>
"""
            
            # Attach both versions
            part1 = MIMEText(text_content, "plain")
            part2 = MIMEText(html_content, "html")
            message.attach(part1)
            message.attach(part2)
            
            # Send email
            print(f"Connecting to SMTP server {self.smtp_server}:{self.smtp_port}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)
            
            print(f"✅ Escalation email sent successfully to {to_email}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            print("❌ Email authentication failed. Check your email and password.")
            print("   For Gmail, you may need to use an App Password:")
            print("   https://support.google.com/accounts/answer/185833")
            return False
            
        except smtplib.SMTPException as e:
            print(f"❌ SMTP error occurred: {e}")
            return False
            
        except Exception as e:
            print(f"❌ Error sending escalation email: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test the email service configuration.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not all([self.sender_email, self.sender_password]):
            print("❌ Email credentials not configured")
            return False
        
        try:
            print(f"Testing connection to {self.smtp_server}:{self.smtp_port}...")
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
            print("✅ Email service connection successful")
            return True
        except Exception as e:
            print(f"❌ Email service connection failed: {e}")
            return False