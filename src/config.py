# config.py - Configuration for tech news digest system
import os

# Data Collection settings
COLLECTION_CONFIG = {
    "rss_url": "https://hnrss.org/classic",
    "jina_api_base": "https://r.jina.ai/",
    "processed_items_file": "processed_items.json",
    "articles_directory": "articles",
    "collection_interval_minutes": 60 * 24  # daily
}

# OpenRouter API Configuration
OPENROUTER_CONFIG = {
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "api_base": "https://openrouter.ai/api/v1",
    "model": "google/gemini-2.0-flash-001",  # "google/gemini-2.0-pro-exp-02-05:free",  # "openrouter/auto", # "deepseek/deepseek-r1:free",
    "max_tokens": 1024,
    "temperature": 0.2,
    "timeout_seconds": 60
}

# User Interest Profile
INTERESTS = {
    # Primary interests - topics that you definitely want to see (more descriptive)
    "primary_interests": [
        "Software architectural approaches",
        "Personal development for knowledge workers like software engineers",
        "Mental resilience and perseverance techniques for professionals"

    ],
    # Excluded interests - topics you explicitly do not want to see
    "excluded_interests": [
        "Cryptocurrency and blockchain applications",
        "Consumer gadget reviews",
        "Tool-specific tutorials and walkthroughs",
        "New programming languages",
        "Anything related to front-end development like CSS, HTML, JavaScript, React, etc.",
        "Anything related to databases",
        "Anything related to hardware from Intel, AMD, Nvidia, etc.",
        "Launch/Review of a open source tool or project",
    ],
}

# Summarization settings
SUMMARIZATION_CONFIG = {
    "max_summary_length": 150,  # Maximum length of summary in words
    "include_key_points": 3,  # Number of key points to extract
}

# Digest compilation settings
DIGEST_CONFIG = {
    "digest_title_format": "Tech News Digest - {date}",
    "output_file_format": "digest_{date}.md",
    "group_by_topics": True,  # Whether to group articles by topic
    "max_topic_clusters": 5  # Maximum number of topic clusters
}

# Logging configuration
LOGGING_CONFIG = {
    "log_file": "tech_digest.log",
    "log_level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    "console_output": True
}

# Evaluation system configuration
EVALUATION_CONFIG = {
    "evaluation_logs_directory": "evaluation_logs",
    "default_days_to_display": 30,
    "required_agreement_rate": 0.8,  # Target agreement rate between model and human
    "include_feedback_examples": True,  # Whether to include human feedback examples in prompts
    "examples_count": 2  # Number of human feedback examples to include
}

FILTERING_CONFIG = {
    "max_articles_per_digest": 10,  # Maximum number of articles to include
    "evaluation_cache_directory": "evaluation_cache"  # Keep cache directory setting
}