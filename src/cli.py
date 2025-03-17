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

def filter_articles():
    """Run the content filtering module to select relevant articles."""
    logger.info("Starting content filtering...")
    
    try:
        from content_filter import filter_articles
        articles_dir = COLLECTION_CONFIG['articles_directory']
        selected_articles = filter_articles(articles_dir)
        logger.info(f"Selected {len(selected_articles)} articles")
        return selected_articles
    except Exception as e:
        logger.error(f"Content filtering failed: {e}")
        return []

def summarize_articles(selected_articles):
    """Run the content summarization module on selected articles."""
    if not selected_articles:
        logger.warning("No articles to summarize")
        return []
    
    logger.info(f"Summarizing {len(selected_articles)} articles...")
    
    try:
        from content_summarizer import summarize_selected_articles
        summarized_articles = summarize_selected_articles(selected_articles)
        logger.info(f"Summarized {len(summarized_articles)} articles")
        return summarized_articles
    except Exception as e:
        logger.error(f"Content summarization failed: {e}")
        return []

def compile_digest(summarized_articles):
    """Run the digest compilation module to create the final digest."""
    if not summarized_articles:
        logger.warning("No summarized articles to compile into a digest")
        return None
    
    logger.info("Compiling digest...")
    
    try:
        from digest_compiler import compile_digest, print_digest_to_console
        digest_file = compile_digest(summarized_articles)
        
        if digest_file:
            logger.info(f"Digest compiled successfully: {digest_file}")
            print_digest_to_console(digest_file)
            return digest_file
        else:
            logger.error("Failed to compile digest")
            return None
    except Exception as e:
        logger.error(f"Digest compilation failed: {e}")
        return None

def run_full_pipeline():
    """Run the complete pipeline from data collection to digest compilation."""
    logger.info("Starting tech news digest pipeline...")
    
    # Step 1: Collect data
    if not collect_data():
        logger.error("Pipeline stopped at data collection stage")
        return False
    
    # Step 2: Filter articles
    selected_articles = filter_articles()
    if not selected_articles:
        logger.error("Pipeline stopped at content filtering stage (no articles selected)")
        return False
    
    # Step 3: Summarize articles
    summarized_articles = summarize_articles(selected_articles)
    if not summarized_articles:
        logger.error("Pipeline stopped at content summarization stage (no articles summarized)")
        return False
    
    # Step 4: Compile digest
    digest_file = compile_digest(summarized_articles)
    if not digest_file:
        logger.error("Pipeline stopped at digest compilation stage")
        return False
    
    logger.info("Pipeline completed successfully!")
    return True

def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(description="Tech News Digest System")
    
    # Define command-line arguments
    parser.add_argument("--collect", action="store_true", help="Collect new articles")
    parser.add_argument("--filter", action="store_true", help="Filter articles based on relevance")
    parser.add_argument("--summarize", action="store_true", help="Summarize selected articles")
    parser.add_argument("--compile", action="store_true", help="Compile the final digest")
    parser.add_argument("--run-all", action="store_true", help="Run the complete pipeline")
    
    args = parser.parse_args()
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Please check your settings.")
        return 1
    
    # Determine what to run
    if args.run_all:
        success = run_full_pipeline()
        return 0 if success else 1
    
    # Individual steps
    if args.collect:
        collect_data()
    
    if args.filter:
        selected_articles = filter_articles()
        print(f"Selected {len(selected_articles)} articles")
    
    if args.summarize:
        # Need to filter first if not already done
        if not args.filter:
            selected_articles = filter_articles()
        summarized_articles = summarize_articles(selected_articles)
    
    if args.compile:
        # Need to filter and summarize first if not already done
        if not args.filter and not args.summarize:
            selected_articles = filter_articles()
            summarized_articles = summarize_articles(selected_articles)
        elif not args.summarize:
            summarized_articles = summarize_articles(selected_articles)
        compile_digest(summarized_articles)
    
    # If no specific arguments were given, show help
    if not (args.collect or args.filter or args.summarize or args.compile or args.run_all):
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())