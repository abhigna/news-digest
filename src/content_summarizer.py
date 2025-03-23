import json
import logging
import requests
import time
from config import SUMMARIZATION_CONFIG, OPENROUTER_CONFIG
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List

# Set up logging
logger = logging.getLogger(__name__)

# Define the structured output model
class ArticleSummary(BaseModel):
    summary: str = Field(description="A concise summary of the article")
    key_points: List[str] = Field(description="Key technical takeaways from the article")
    technical_details: List[str] = Field(description="Specific technologies, frameworks, or technical specifications mentioned")

def summarize_article(article_content, article_metadata, evaluation_info):
    """
    Generate a concise summary of an article using LLM.
    
    Args:
        article_content: The full content of the article
        article_metadata: Metadata about the article (title, date, etc.)
        evaluation_info: Information about article evaluation (topics, pass/fail)
        
    Returns:
        dict: Contains summary, key points, and technical details
    """
    # Extract relevant metadata
    title = article_metadata.get("title", "")
    url = article_metadata.get("link", "")
    author = article_metadata.get("creator", "Unknown")
    
    # Get topics from evaluation info
    topics = evaluation_info.get("main_topics", [])
    topics_str = ", ".join(topics) if topics else "technology"
    
    # Truncate content if it's too long to fit in LLM context
    max_content_chars = 6000
    truncated_content = article_content[:max_content_chars] + "..." if len(article_content) > max_content_chars else article_content
    
    key_points_count = SUMMARIZATION_CONFIG["include_key_points"]
    max_length = SUMMARIZATION_CONFIG["max_summary_length"]
    
    prompt = f"""
You are a technical content summarizer for a tech news digest aimed at software engineers and developers.

ARTICLE TO SUMMARIZE:
Title: {title}
URL: {url}
Author: {author}
Topics: {topics_str}
Content: {truncated_content}

TASK:
Create a concise technical summary of this article with these components:

1. Summary: A {max_length}-word summary that captures the main technological points and significance to developers. Focus on technical details, tools, frameworks, or concepts mentioned.

2. Key Points: Extract exactly {key_points_count} key technical takeaways from the article, formatted as bullet points.

3. Technical Details: If the article mentions specific technologies, programming languages, frameworks, APIs, or technical specifications, list them briefly.
"""

    # Initialize OpenAI client with OpenRouter base URL and API key
    base_client = OpenAI(
        base_url=OPENROUTER_CONFIG['api_base'],
        api_key=OPENROUTER_CONFIG['api_key']
    )
    
    # Patch the client with instructor
    client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)
    
    try:
        # Use instructor to get structured output
        result = client.chat.completions.create(
            model=OPENROUTER_CONFIG["model"],
            response_model=ArticleSummary,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=OPENROUTER_CONFIG["max_tokens"],
            temperature=OPENROUTER_CONFIG["temperature"],
            timeout=OPENROUTER_CONFIG['timeout_seconds']
        )
        
        # Convert Pydantic model to dict
        return result.model_dump()
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {
            'summary': f"Error: {str(e)}",
            'key_points': [],
            'technical_details': []
        }
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return {
            'summary': f"Error: {str(e)}",
            'key_points': [],
            'technical_details': []
        }

def summarize_selected_articles(selected_articles):
    """
    Generate summaries for a list of selected articles.
    
    Args:
        selected_articles: List of articles with metadata and relevance info
        
    Returns:
        list: Articles with added summary information
    """
    summarized_articles = []
    
    for article in selected_articles:
        file_path = article['file_path']
        metadata = article['metadata']
        evaluation = article['evaluation']
        
        try:
            # Load the full article content
            with open(file_path, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            
            content = article_data.get('content', '')
            
            # Skip if no content
            if not content:
                logger.warning(f"Skipping article with no content: {file_path}")
                continue
            
            # Generate summary
            logger.info(f"Summarizing article: {metadata.get('title', 'Unknown')}")
            summary_result = summarize_article(content, metadata, evaluation)
            
            # Add summary to article data
            article_with_summary = article.copy()
            article_with_summary['summary'] = summary_result
            
            summarized_articles.append(article_with_summary)
            
        except Exception as e:
            logger.error(f"Error summarizing article {file_path}: {e}")
    
    logger.info(f"Summarized {len(summarized_articles)} articles")
    
    return summarized_articles

if __name__ == "__main__":
    # Example standalone usage
    from content_filter import filter_articles
    from config import COLLECTION_CONFIG
    
    articles_dir = COLLECTION_CONFIG['articles_directory']
    selected = filter_articles(articles_dir)
    summarized = summarize_selected_articles(selected)
    
    print(f"Summarized {len(summarized)} articles")
    for article in summarized:
        print(f"- {article['metadata'].get('title', 'Unknown')}")
        print(f"  Summary: {article['summary']['summary'][:100]}...")