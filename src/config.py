# Configuration for Tech News Digest System
import os

# Data Collection settings
COLLECTION_CONFIG = {
    "rss_url": "https://hnrss.org/classic",
    "jina_api_base": "https://r.jina.ai/",
    "processed_items_file": "processed_items.json",
    "articles_directory": "articles",
    "collection_interval_minutes": 60 * 24 # daily
}

# OpenRouter API Configuration
OPENROUTER_CONFIG = {
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "api_base": "https://openrouter.ai/api/v1",
    "model": "google/gemini-2.0-pro-exp-02-05:free", # "openrouter/auto", # "deepseek/deepseek-r1:free",
    "max_tokens": 1024,
    "temperature": 0.2,
    "timeout_seconds": 60
}

# User Interest Profile
# This determines which articles are relevant to the user
INTERESTS = {
    "primary_topics": [
        "machine learning",
        "artificial intelligence",
        "large language models",
        "Python programming",
        "software engineering",
        "distributed systems",
        "system design",
        "backend development"
    ],
    "secondary_topics": [
        "cloud computing",
        "databases",
        "data engineering",
        "web development",
        "API design",
        "DevOps",
        "microservices",
        "containerization"
    ],
    "excluded_topics": [
        "cryptocurrency",
        "blockchain",
        "NFTs",
        "web3"
    ]
}

# Content Filtering settings
FILTERING_CONFIG = {
    "relevance_threshold": 0.7,  # Minimum relevance score to include article (0-1)
    "max_articles_per_digest": 10,  # Maximum number of articles to include
    "recency_boost_days": 2  # Articles published within this many days get a score boost
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