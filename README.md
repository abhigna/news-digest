Yes, you can make your README more navigatable on GitHub by adding a table of contents and using anchor links. This will help users quickly jump to different sections of the document. Here's how you can enhance your README:

# Tech News Digest System

A personal tool for creating customized tech news digests based on your specific interests.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Setup Instructions](#setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Initial Setup](#initial-setup)
- [Usage](#usage)
  - [Collecting Articles](#collecting-articles)
  - [Compiling a Digest](#compiling-a-digest)
  - [Using the Evaluation App](#using-the-evaluation-app)
- [Customization](#customization)
  - [Configuring Interests](#configuring-interests)
  - [Scheduling Automatic Updates](#scheduling-automatic-updates)
  - [Model Configuration](#model-configuration)
- [Output](#output)

## Overview

This repository contains a personal tool for creating customized tech news digests that match your specific interests. While it currently uses Hacker News as its source, it's designed to be expandable to other news sources.

## System Architecture

```
src
├── content_filter.py       # Filters articles based on user interests
├── cli.py                  # Command-line interface for the application
├── config.py               # Configuration settings for the system
├── hn_data_collector.py    # Collects articles from Hacker News
├── content_summarizer.py   # Generates summaries for selected articles
├── eval_app.py             # Web app for evaluating and improving filtering
├── digest_compiler.py      # Compiles articles into a final digest
└── evaluation_system.py    # Tracks filtering performance and agreement
```

### Component Overview

- **hn_data_collector.py**: Fetches articles from Hacker News RSS feed and uses Jina API to extract content
- **content_filter.py**: Uses LLM to evaluate articles against your interest profile
- **content_summarizer.py**: Creates concise summaries of filtered articles
- **digest_compiler.py**: Generates a formatted Markdown digest, optionally grouping by topics
- **evaluation_system.py**: Tracks how well the filtering matches your preferences
- **eval_app.py**: Streamlit app for reviewing filtering decisions and improving the system
- **config.py**: Central configuration file for all system components
- **cli.py**: Command-line interface to run various pipeline stages

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- An OpenRouter API key (for LLM services)

### Initial Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd news-digest
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key:
   
   Either create a `.env` file:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
   
   Or set it as an environment variable:
   ```bash
   export OPENROUTER_API_KEY=your_api_key_here
   ```

4. Customize your interests in `config.py`:
   - Edit the `INTERESTS` section to specify your interests
   - Add topics to `primary_interests` that you want to see
   - Add topics to `excluded_interests` that you want to filter out

## Usage

### Collecting Articles

To fetch new articles from Hacker News (recommended to run daily via cron):

```bash
python src/cli.py --collect
```

### Filtering Articles

To filter the collected articles based on your interests:

```bash
python src/cli.py --filter
```

You can limit to recent articles with the `--days` parameter:

```bash
python src/cli.py --filter --days 1
```

### Summarizing Articles

To generate summaries for filtered articles:

```bash
python src/cli.py --summarize
```

### Compiling a Digest

To create a final digest as a Markdown file:

```bash
python src/cli.py --compile
```

For recent articles only:

```bash
python src/cli.py --compile --days 1
```

### Running the Full Pipeline

To execute all steps (collect, filter, summarize, compile) at once:

```bash
python src/cli.py --run-all
```

### Using the Evaluation App

To review article filtering decisions and improve the system:

```bash
streamlit run src/eval_app.py
```

The evaluation app allows you to:
- View filtered articles
- Provide feedback on filtering decisions
- Review statistics on filter accuracy
- Add article exceptions
- Refine your interest profile

## Customization

### Configuring Interests

Edit the `INTERESTS` section in `config.py` to customize filtering:

```python
INTERESTS = {
    "primary_interests": [
        "Software architectural approaches",
        "Personal development for knowledge workers"
    ],
    "excluded_interests": [
        "Cryptocurrency and blockchain applications",
        "Consumer gadget reviews"
    ]
}
```

### Scheduling Automatic Updates

To automate daily collection of articles, set up a cron job:

```
# Run at 6 AM every day
0 6 * * * cd /path/to/news-digest && python src/cli.py --collect

# Generate digest at 7 AM every day with articles from last 24 hours
0 7 * * * cd /path/to/news-digest && python src/cli.py --compile --days 1
```

### Model Configuration

The system uses OpenRouter for access to different AI models. You can configure the model in `config.py`:

```python
OPENROUTER_CONFIG = {
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "api_base": "https://openrouter.ai/api/v1",
    "model": "google/gemini-2.0-flash-001",  # Change model here
    "max_tokens": 1024,
    "temperature": 0.2,
    "timeout_seconds": 60
}
```

## Output

The system generates a Markdown digest file (`digest_YYYY-MM-DD.md`) that can be viewed in any Markdown editor or converted to other formats.