import os
import dotenv
dotenv.load_dotenv()

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
    "model": "google/gemini-2.0-flash-lite-001", # "google/gemini-2.0-flash-001",  # "google/gemini-2.0-pro-exp-02-05:free",  # "openrouter/auto", # "deepseek/deepseek-r1:free",
    "max_tokens": 1024,
    "temperature": 0.2,
    "timeout_seconds": 60
}

# Model Critique Configuration
CRITIQUE_CONFIG = {
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "api_base": "https://openrouter.ai/api/v1",
    "model": "google/gemini-2.5-pro-exp-03-25:free",  # "anthropic/claude-3-7-sonnet:2024-08-08",
    "max_tokens": 2048,
    "temperature": 0.5,
    "timeout_seconds": 120
}

# User Interest Profile
INTERESTS = {
    # Primary interests - topics that you definitely want to see (more descriptive)
    "primary_interests": [
        "AI-assisted programming and development workflows",
        "Impact of AI on society and the economy",
        "Personal stories from seasoned professionals in software engineering",
    ],
    # Excluded interests - topics you explicitly do not want to see
    "excluded_interests": [
        "Cryptocurrency and blockchain applications",
        "Consumer gadget reviews",
        "Anything related to front-end development like CSS, HTML, JavaScript, React, etc.",
    ],
}

# Summarization settings
# Let's assume these would be added to config.py
SUMMARIZATION_CONFIG = {
    "max_summary_length": 150,
    "llm_cache": "llm_cache/summaries",
    "use_cache": True,
    # TODO: "include_feedback_examples": True,
    "examples_count": 2
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
JUDGE_CONFIG = {
    "llm_trace_logs": "llm_trace_logs",
    "default_days_to_display": 30,
    "human_feedback_file": "human_feedback.json"
}

FILTERING_CONFIG = {
    "max_articles_per_digest": 10,  # Maximum number of articles to include
    "llm_cache": "llm_cache/filtering",  # Keep cache directory setting
    "include_feedback_examples": True,
    "examples_count": 10,
}