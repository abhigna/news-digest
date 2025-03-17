import os
import json
import logging
import requests
from datetime import datetime, timedelta
import time
from config import INTERESTS, FILTERING_CONFIG, OPENROUTER_CONFIG
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tech_digest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("content_filter")

# Define the structured output model
class RelevanceScore(BaseModel):
    relevance_score: float = Field(
        description="A score from 0.0 to 1.0 indicating how relevant the article is to user interests",
        ge=0.0,
        le=1.0
    )
    main_topics: List[str] = Field(
        description="The main topics covered in the article"
    )
    reasoning: str = Field(
        description="Brief explanation for why this relevance score was assigned"
    )

def calculate_relevance_score(article_content, article_metadata):
    """
    Use LLM to calculate relevance score based on user interests.
    
    Args:
        article_content: The full content of the article
        article_metadata: Metadata about the article (title, date, etc.)
        
    Returns:
        dict: Contains relevance score (0-1), reasoning, and topics
    """
    # Prepare prompt for the LLM
    primary_topics = ", ".join(INTERESTS["primary_topics"])
    secondary_topics = ", ".join(INTERESTS["secondary_topics"])
    excluded_topics = ", ".join(INTERESTS["excluded_topics"])
    
    title = article_metadata.get("title", "")
    
    # Truncate content if it's too long to fit in LLM context
    max_content_chars = 6000  # Reasonable limit to keep context size manageable
    truncated_content = article_content[:max_content_chars] + "..." if len(article_content) > max_content_chars else article_content
    
    prompt = f"""
You are a content relevance scoring system for a tech news digest.

USER INTERESTS:
Primary Topics: {primary_topics}
Secondary Topics: {secondary_topics}
Excluded Topics: {excluded_topics}

ARTICLE TO EVALUATE:
Title: {title}
Content: {truncated_content}

TASK:
1. Identify the main topics covered in this article.
2. Determine how relevant this article is to the user's interests on a scale from 0.0 to 1.0.
   - Score 0.9-1.0: Extremely relevant (matches multiple primary interests)
   - Score 0.7-0.8: Highly relevant (matches a primary interest well)
   - Score 0.5-0.6: Moderately relevant (matches secondary interests or partially matches primary)
   - Score 0.3-0.4: Slightly relevant (tangentially related to interests)
   - Score 0.0-0.2: Not relevant (unrelated to interests or focuses on excluded topics)
3. Provide brief reasoning for your score.
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
            response_model=RelevanceScore,
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
            'relevance_score': 0.0,
            'main_topics': [],
            'reasoning': f"API request failed: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error in relevance calculation: {e}")
        return {
            'relevance_score': 0.0,
            'main_topics': [],
            'reasoning': f"Error processing response: {str(e)}"
        }

def apply_recency_boost(relevance_score, pub_date_str):
    """Apply a boost to the relevance score for recent articles."""
    try:
        # Parse the publication date
        # Handle various date formats
        for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
            try:
                pub_date = datetime.strptime(pub_date_str, fmt)
                break
            except ValueError:
                continue
        else:
            logger.warning(f"Could not parse date format: {pub_date_str}")
            return relevance_score  # Return unchanged if date parsing fails
        
        # Calculate how recent the article is
        now = datetime.now()
        if pub_date.tzinfo is not None:
            now = datetime.now(pub_date.tzinfo)
        
        days_old = (now - pub_date).days
        
        # Apply boost if within the recency window
        if days_old <= FILTERING_CONFIG['recency_boost_days']:
            boost_factor = 1.0 - (days_old / (FILTERING_CONFIG['recency_boost_days'] * 2))
            
            # Ensure we don't exceed 1.0
            return min(relevance_score * (1.0 + boost_factor), 1.0)
        
        return relevance_score
        
    except Exception as e:
        logger.error(f"Error applying recency boost: {e}")
        return relevance_score  # Return unchanged if any error occurs

def filter_articles(articles_directory):
    """
    Filter articles based on relevance to user interests.
    
    Args:
        articles_directory: Directory containing article JSON files
        
    Returns:
        list: Selected articles with relevance scores
    """
    # Get all article files
    article_files = [f for f in os.listdir(articles_directory) if f.endswith('.json')]
    logger.info(f"Found {len(article_files)} articles to filter")
    
    filtered_articles = []
    
    for article_file in article_files:
        file_path = os.path.join(articles_directory, article_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
            
            # Extract content and metadata
            content = article_data.get('content', '')
            metadata = article_data.get('metadata', {})
            
            # Skip if no content
            if not content:
                logger.warning(f"Skipping article with no content: {file_path}")
                continue
            
            # Calculate relevance score
            logger.info(f"Evaluating relevance of article: {metadata.get('title', 'Unknown')}")
            relevance_result = calculate_relevance_score(content, metadata)
            
            # Apply recency boost
            initial_score = relevance_result['relevance_score']
            pub_date = metadata.get('pubDate', '')
            if pub_date:
                relevance_result['relevance_score'] = apply_recency_boost(initial_score, pub_date)
                if relevance_result['relevance_score'] > initial_score:
                    relevance_result['reasoning'] += f" (Score boosted from {initial_score:.2f} due to recency)"
            
            # Add to filtered list if above threshold
            if relevance_result['relevance_score'] >= FILTERING_CONFIG['relevance_threshold']:
                filtered_articles.append({
                    'file_path': file_path,
                    'metadata': metadata,
                    'relevance': relevance_result
                })
                logger.info(f"Article accepted with score {relevance_result['relevance_score']:.2f}: {metadata.get('title', 'Unknown')}")
            else:
                logger.info(f"Article rejected with score {relevance_result['relevance_score']:.2f}: {metadata.get('title', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error processing article {file_path}: {e}")
    
    # Sort by relevance score (descending) and limit to max articles
    filtered_articles.sort(key=lambda x: x['relevance']['relevance_score'], reverse=True)
    selected_articles = filtered_articles[:FILTERING_CONFIG['max_articles_per_digest']]
    
    logger.info(f"Selected {len(selected_articles)} articles out of {len(article_files)} total")
    
    return selected_articles

if __name__ == "__main__":
    # Example standalone usage
    from config import COLLECTION_CONFIG
    articles_dir = COLLECTION_CONFIG['articles_directory']
    selected = filter_articles(articles_dir)
    print(f"Selected {len(selected)} articles")
    for article in selected:
        print(f"- {article['metadata'].get('title', 'Unknown')} (Score: {article['relevance']['relevance_score']:.2f})")