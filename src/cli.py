#!/usr/bin/env python3
"""
Tech News Digest System

A system that collects, filters, summarizes, and compiles tech news articles 
based on user interests.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime

import dotenv

dotenv.load_dotenv()

# Import configuration
from config import COLLECTION_CONFIG, LOGGING_CONFIG

# Set up logging
log_level = getattr(logging, LOGGING_CONFIG["log_level"])
handlers = [logging.FileHandler(LOGGING_CONFIG["log_file"])]
if LOGGING_CONFIG["console_output"]:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger("tech_digest")

def validate_config():
    """Validate configuration and environment."""
    # Check for OpenRouter API key
    
    # Check for articles directory
    articles_dir = COLLECTION_CONFIG['articles_directory']
    if not os.path.exists(articles_dir):
        logger.info(f"Creating articles directory: {articles_dir}")
        try:
            os.makedirs(articles_dir)
        except Exception as e:
            logger.error(f"Failed to create articles directory: {e}")
            return False
    
    return True

def collect_data():
    """Run the data collection module to fetch new articles."""
    logger.info("Starting data collection...")
    
    try:
        from hn_data_collector import main as collector_main
        collector_main()
        logger.info("Data collection completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

def filter_articles(use_cache=True, days=None):
    """Run the content filtering module to select relevant articles."""
    logger.info("Starting content filtering...")
    
    try:
        # Import the required classes and configurations
        from llm_gateway import LlmGateway
        from content_filter import ContentFilter
        from judge_system import JudgeSystem
        from config import FILTERING_CONFIG, OPENROUTER_CONFIG, INTERESTS, COLLECTION_CONFIG, JUDGE_CONFIG
        
        # Initialize the LLM gateway
        llm_gateway = LlmGateway(
            llm_config=OPENROUTER_CONFIG,
            use_case='content_filter', 
            cache_dir=FILTERING_CONFIG.get('llm_cache', 'llm_cache')
        )
        
        # Initialize the judge system (for feedback examples)
        judge_system = JudgeSystem(JUDGE_CONFIG)
        
        # Initialize the content filter
        content_filter = ContentFilter(
            filter_config=FILTERING_CONFIG,
            llm_gateway=llm_gateway,
            interests=INTERESTS,
            judge_system=judge_system
        )
        
        # Run the filtering
        articles_dir = COLLECTION_CONFIG['articles_directory']
        selected_articles = content_filter.filter_articles(articles_dir, use_cache=use_cache, days=days)
        
        logger.info(f"Selected {len(selected_articles)} articles")
        return selected_articles
    except Exception as e:
        logger.error(f"Content filtering failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def summarize_articles(selected_articles):
    """Run the content summarization module on selected articles."""
    if not selected_articles:
        logger.warning("No articles to summarize")
        return []
    
    logger.info(f"Summarizing {len(selected_articles)} articles...")
    
    try:
        # Import required classes and configurations
        from llm_gateway import LlmGateway
        from content_summarizer import ContentSummarizer
        from judge_system import JudgeSystem
        from config import SUMMARIZATION_CONFIG, OPENROUTER_CONFIG, JUDGE_CONFIG
        
        # Initialize the LLM gateway
        llm_gateway = LlmGateway(
            llm_config=OPENROUTER_CONFIG,
            use_case='content_summarizer', 
            cache_dir=SUMMARIZATION_CONFIG.get('llm_cache', 'llm_cache')
        )
        
        # Initialize the judge system
        judge_system = JudgeSystem(JUDGE_CONFIG)
        
        # Initialize the content summarizer
        content_summarizer = ContentSummarizer(
            summarization_config=SUMMARIZATION_CONFIG,
            llm_gateway=llm_gateway,
            judge_system=judge_system
        )
        
        # Run the summarization
        summarized_articles = content_summarizer.summarize_selected_articles(selected_articles)
        logger.info(f"Summarized {len(summarized_articles)} articles")
        return summarized_articles
    except Exception as e:
        logger.error(f"Content summarization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def compile_digest(summarized_articles):
    """Run the digest compilation module to create the final digest."""
    if not summarized_articles:
        logger.warning("No summarized articles to compile into a digest")
        return None
    
    logger.info("Compiling digest...")
    
    try:
        # Import required classes and configurations
        from digest_compiler import DigestCompiler
        from judge_system import JudgeSystem
        from config import DIGEST_CONFIG, JUDGE_CONFIG
        
        # Initialize the judge system
        judge_system = JudgeSystem(JUDGE_CONFIG)
        
        # Initialize the digest compiler
        digest_compiler = DigestCompiler(
            digest_config=DIGEST_CONFIG,
            judge_system=judge_system
        )
        
        # Compile the digest
        digest_file = digest_compiler.compile_digest(summarized_articles)
        
        if digest_file:
            logger.info(f"Digest compiled successfully: {digest_file}")
            digest_compiler.print_digest_to_console(digest_file)
            return digest_file
        else:
            logger.error("Failed to compile digest")
            return None
    except Exception as e:
        logger.error(f"Digest compilation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def send_email(digest_file, to_email):
    """Send the digest via email."""
    if not digest_file:
        logger.warning("No digest file to send")
        return False
        
    logger.info(f"Sending digest via email to {to_email}...")
    
    try:
        # Import the EmailSender class
        from email_sender import EmailSender
        
        # Initialize the email sender
        email_sender = EmailSender()
        
        # Send the digest
        result = email_sender.send_digest(digest_file, to_email)
        
        if result:
            logger.info(f"Digest sent successfully to {to_email}")
            return True
        else:
            logger.error(f"Failed to send digest to {to_email}")
            return False
    except Exception as e:
        logger.error(f"Email sending failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Tech News Digest System")
    
    # Define command-line arguments
    parser.add_argument("--collect", action="store_true", help="Collect new articles")
    parser.add_argument("--filter", action="store_true", help="Filter articles based on relevance")
    parser.add_argument("--summarize", action="store_true", help="Summarize selected articles")
    parser.add_argument("--compile", action="store_true", help="Compile the final digest")
    parser.add_argument("--no-cache", action="store_true", help="Disable content filtering cache")
    parser.add_argument("--days", type=int, default=None, 
                        help="Process only articles from the last X days (default: all articles)")
    parser.add_argument("--email", type=str, default=None,
                        help="Email address to send the digest to")
    
    args = parser.parse_args()
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Please check your settings.")
        return 1
    
    # Get cache and days settings
    use_cache = not args.no_cache
    days_filter = args.days
    
    # Variables to track state
    selected_articles = []
    summarized_articles = []
    digest_file = None
    
    # Individual steps
    if args.collect:
        collect_data()
    
    if args.filter:
        selected_articles = filter_articles(use_cache=use_cache, days=days_filter)
        print(f"Selected {len(selected_articles)} articles")
    
    if args.summarize:
        # Need to filter first if not already done
        if not args.filter:
            selected_articles = filter_articles(use_cache=use_cache, days=days_filter)
        summarized_articles = summarize_articles(selected_articles)
    
    if args.compile:
        # Need to filter and summarize first if not already done
        if not args.filter and not args.summarize:
            selected_articles = filter_articles(use_cache=use_cache, days=days_filter)
            summarized_articles = summarize_articles(selected_articles)
        elif not args.summarize:
            summarized_articles = summarize_articles(selected_articles)
        digest_file = compile_digest(summarized_articles)
    
    # Send email if requested
    if args.email:
        if not digest_file and args.compile:
            logger.error("No digest file to send")
        else:
            # If no digest was compiled in this run, find the latest one
            if not digest_file:
                import glob
                digest_files = glob.glob("digests/digest_*.md")
                if digest_files:
                    digest_file = max(digest_files, key=os.path.getctime)
                    logger.info(f"Using latest digest file: {digest_file}")
                else:
                    logger.error("No digest files found")
            
            # Send the email
            if digest_file:
                send_email(digest_file, args.email)
    
    # If no specific arguments were given, show help
    if not (args.collect or args.filter or args.summarize or args.compile or args.email):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())