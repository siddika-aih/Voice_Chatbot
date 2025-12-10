"""Email sending tool"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from typing import Dict, Any

class EmailTool:
    """Send emails via SMTP"""
    
    # Tool declaration for Gemini
    DECLARATION = {
        "name": "send_email",
        "description": "Send an email to a recipient. Use this when user asks to send email, compose message, or email someone.",
        "parameters": {
            "type": "object",
            "properties": {
                "to_email": {
                    "type": "string",
                    "description": "Recipient's email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                }
            },
            "required": ["to_email", "subject", "body"]
        }
    }
    
    def __init__(self):
        # Configure with your SMTP settings
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD")
    
    async def execute(self, to_email: str, subject: str, body: str) -> Dict[str, Any]:
        """
        Send email
        
        Returns:
            Dict with success status and message
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return {
                "success": True,
                "message": f"Email sent successfully to {to_email}",
                "to": to_email,
                "subject": subject
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to send email: {str(e)}",
                "error": str(e)
            }
