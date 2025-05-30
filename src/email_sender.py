import os
import logging
import markdown
import resend

# Set up logging
logger = logging.getLogger("email_sender")

class EmailSender:
    """
    Email sender module that converts markdown digests to HTML and sends them via email.
    """
    
    def __init__(self, from_email=os.environ.get("EMAIL_FROM")):
        """
        Initialize the email sender.
        
        Args:
            from_email: The email address to send from
        """
        self.from_email = from_email
        
        # Get API key from environment
        api_key = os.environ.get("RESEND_API_KEY")
        if not api_key:
            logger.error("RESEND_API_KEY environment variable is not set")
            raise ValueError("RESEND_API_KEY environment variable is not set")
        
        # Set the API key for Resend
        resend.api_key = api_key
    
    def markdown_to_html(self, markdown_file):
        """
        Convert a markdown file to HTML.
        
        Args:
            markdown_file: Path to the markdown file
            
        Returns:
            str: HTML content
        """
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                md_content = f.read()
            
            # Convert markdown to HTML with enhanced extensions
            html_content = markdown.markdown(
                md_content, 
                extensions=[
                    'extra',
                    'codehilite',
                    'sane_lists',
                    'nl2br',  # Convert newlines to <br> tags
                    'tables'  # Better table support
                ],
                output_format='html5'  # Use HTML5 output format
            )
            
            # Add improved styling with specific list styling
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        line-height: 1.6; 
                        max-width: 800px; 
                        margin: 0 auto; 
                        padding: 20px; 
                        color: #333;
                    }}
                    h1, h2, h3 {{ color: #333; margin-top: 1.5em; margin-bottom: 0.5em; }}
                    a {{ color: #0066cc; text-decoration: none; }}
                    a:hover {{ text-decoration: underline; }}
                    code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 4px; }}
                    hr {{ border: 0; border-top: 1px solid #eee; margin: 20px 0; }}
                    
                    /* Improved list styling */
                    ul, ol {{ 
                        padding-left: 20px; 
                        margin: 1em 0; 
                    }}
                    ul li, ol li {{ 
                        margin-bottom: 0.5em; 
                        padding-left: 5px;
                    }}
                    ul {{ 
                        list-style-type: disc; 
                    }}
                    ul ul {{ 
                        list-style-type: circle; 
                    }}
                    ul ul ul {{ 
                        list-style-type: square; 
                    }}
                    
                    /* Specific email client fixes */
                    /* For Outlook */
                    .outlook-list {{
                        margin-left: 30px !important;
                    }}
                    /* For Gmail */
                    .gmail-fix {{
                        display: block !important;
                    }}
                </style>
            </head>
            <body>
                <!-- Email client specific wrapper for lists -->
                <div class="gmail-fix outlook-list">
                    {html_content}
                </div>
            </body>
            </html>
            """
            
            return styled_html
            
        except Exception as e:
            logger.error(f"Error converting markdown to HTML: {e}")
            return None
    
    def send_digest(self, digest_file, to_email):
        """
        Send the digest as an email.
        
        Args:
            digest_file: Path to the digest markdown file
            to_email: Recipient email address
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract date from filename for subject line
            filename = os.path.basename(digest_file)
            date_str = filename.split('.')[0].replace('digest_', '')
            subject = f"Tech News Digest - {date_str}"
            
            # Convert markdown to HTML
            html_content = self.markdown_to_html(digest_file)
            if not html_content:
                logger.error("Failed to convert digest to HTML")
                return False
            
            # Prepare email parameters
            params = {
                "from": f"Tech News Digest <{self.from_email}>",
                "to": [to_email],
                "subject": subject,
                "html": html_content,
            }
            
            # Send the email
            logger.info(f"Sending digest to {to_email}")
            email_response = resend.Emails.send(params)
            logger.info(f"Email sent successfully. Response: {email_response}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False