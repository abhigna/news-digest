import os
import json
import logging
import requests
from datetime import datetime
import time
import hashlib
from config import INTERESTS, FILTERING_CONFIG, OPENROUTER_CONFIG, EVALUATION_CONFIG
from evaluation_system import EvaluationSystem
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# Define the structured output model with binary pass/fail
class ContentEvaluation(BaseModel):
    pass_filter: bool = Field(
        description="Whether the article passes the filter based on relevance to user interests"
    )
    main_topics: List[str] = Field(
        description="The main topics covered in the article (be specific and descriptive)"
    )
    reasoning: str = Field(
        description="Brief explanation for why this article passes or fails the filter"
    )
    specific_interests_matched: List[str] = Field(
        description="The specific user interests that this article matches",
        default=[]
    )

def evaluate_content(article_content, article_metadata):
    """
    Use LLM to evaluate content with a binary pass/fail approach.
    
    Args:
        article_content: The full content of the article
        article_metadata: Metadata about the article (title, date, etc.)
        
    Returns:
        dict: Contains pass/fail decision, reasoning, and topics
    """
    # Prepare prompt for the LLM with detailed interest descriptions
    primary_interests = INTERESTS.get("primary_interests", [])
    excluded_interests = INTERESTS.get("excluded_interests", [])
    
    # Format interests in a more descriptive way
    primary_interests_text = "\n".join([f"- {interest}" for interest in primary_interests])
    excluded_interests_text = "\n".join([f"- {interest}" for interest in excluded_interests])
    
    title = article_metadata.get("title", "")
    source = article_metadata.get("source", "")
    
    # Truncate content if it's too long to fit in LLM context
    max_content_chars = 6000
    truncated_content = article_content[:max_content_chars] + "..." if len(article_content) > max_content_chars else article_content
    
    prompt = f"""
You are a content filtering system for a personalized tech news digest.

USER INTERESTS DESCRIPTION:
The user is specifically interested in:
{primary_interests_text}

The user is NOT interested in:
{excluded_interests_text}

ARTICLE TO EVALUATE:
Title: {title}
Source: {source}
Content: {truncated_content}

TASK:
1. Identify the specific main topics covered in this article (be detailed and descriptive).
2. Make a binary decision: Does this article PASS or FAIL the filter?
   - PASS: The article significantly matches one or more of the user's primary interests
   - FAIL: The article doesn't match user interests OR primarily focuses on excluded topics
3. Provide reasoning for your decision, explaining why this article would or would not be valuable to the user.
4. List the specific user interests matched (if any).

This evaluation will be used for both filtering and human validation, so be thorough in your reasoning.
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
            response_model=ContentEvaluation,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=OPENROUTER_CONFIG["max_tokens"],
            temperature=OPENROUTER_CONFIG["temperature"],
            timeout=OPENROUTER_CONFIG['timeout_seconds']
        )
        
        # Convert Pydantic model to dict
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Error in content evaluation: {e}")
        return {
            'pass_filter': False,
            'main_topics': [],
            'reasoning': f"Error processing article: {str(e)}",
            'specific_interests_matched': []
        }

def get_interests_hash():
    """
    Generate a hash representing the current interests configuration.
    This allows us to detect changes in the interests.
    
    Returns:
        str: Hash of the current interests configuration
    """
    # Convert interests to a string representation for hashing
    interests_str = json.dumps(INTERESTS, sort_keys=True)
    return hashlib.md5(interests_str.encode()).hexdigest()

def get_cached_evaluation(article_path, interests_hash):
    """
    Check if we have a cached evaluation for this article with current interests.
    
    Args:
        article_path: Path to the article file
        interests_hash: Hash of current interests configuration
        
    Returns:
        dict or None: Cached evaluation result or None if not found
    """
    cache_dir = FILTERING_CONFIG.get('evaluation_cache_directory', 'evaluation_cache')
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a unique ID for this article
    article_id = os.path.basename(article_path)
    cache_path = os.path.join(cache_dir, f"{article_id}_{interests_hash}.json")
    
    # Check if cached result exists
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
            return cached_result
        except Exception as e:
            logger.warning(f"Failed to load cached evaluation: {e}")
    
    return None

def save_evaluation_cache(article_path, interests_hash, evaluation_result):
    """
    Save evaluation result to cache.
    
    Args:
        article_path: Path to the article file
        interests_hash: Hash of current interests configuration
        evaluation_result: The evaluation result to cache
    """
    cache_dir = FILTERING_CONFIG.get('evaluation_cache_directory', 'evaluation_cache')
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a unique ID for this article
    article_id = os.path.basename(article_path)
    cache_path = os.path.join(cache_dir, f"{article_id}_{interests_hash}.json")
    
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=2)
        logger.debug(f"Cached evaluation saved to {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save evaluation cache: {e}")

def filter_articles(articles_directory, use_cache=True, days=None):
    """
    Filter articles based on binary pass/fail evaluation.
    
    Args:
        articles_directory: Directory containing article JSON files
        use_cache: Whether to use cached evaluations when available
        days: Only process articles from the last X days (None means all articles)
        
    Returns:
        list: filtered_articles containing articles that passed the filter
    """
    # Initialize the evaluation system
    eval_system = EvaluationSystem(EVALUATION_CONFIG)
    
    # Get all article files
    article_files = [f for f in os.listdir(articles_directory) if f.endswith('.json')]
    logger.info(f"Found {len(article_files)} articles to filter")
    
    filtered_articles = []
    cache_hits = 0
    
    # Generate interests hash for caching
    interests_hash = get_interests_hash() if use_cache else None
    
    # Calculate cutoff date if days parameter is provided
    cutoff_date = None
    if days is not None:
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        logger.info(f"Filtering only articles from the last {days} days")
    
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
            
            # Apply days filter if specified
            if cutoff_date is not None:
                pub_date_str = metadata.get('pubDate', '')
                if not pub_date_str:
                    logger.warning(f"Article has no pubDate, skipping days filter: {file_path}")
                else:
                    # Try to parse the date in various formats
                    article_timestamp = None
                    for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                        try:
                            pub_date = datetime.strptime(pub_date_str, fmt)
                            article_timestamp = pub_date.timestamp()
                            break
                        except ValueError:
                            continue
                    
                    # Raise error if we couldn't parse the date
                    if article_timestamp is None:
                        error_msg = f"Could not parse pubDate '{pub_date_str}' for article: {metadata.get('title', 'Unknown')}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    
                    # Skip article if older than cutoff date
                    if article_timestamp < cutoff_date:
                        logger.info(f"Skipping article older than {days} days: {metadata.get('title', 'Unknown')}")
                        continue
            
            # Check cache first if enabled
            evaluation_result = None
            is_cached = False
            if use_cache:
                evaluation_result = get_cached_evaluation(file_path, interests_hash)
                if evaluation_result:
                    logger.info(f"Using cached evaluation for: {metadata.get('title', 'Unknown')}")
                    cache_hits += 1
                    is_cached = True
            
            # If not in cache, evaluate content
            if not evaluation_result:
                logger.info(f"Evaluating article: {metadata.get('title', 'Unknown')}")
                evaluation_result = evaluate_content(content, metadata)
                
                # Cache the evaluation if caching is enabled
                if use_cache:
                    save_evaluation_cache(file_path, interests_hash, evaluation_result)
            
            # Log evaluation to the evaluation system
            item_id = os.path.basename(file_path).split('.')[0]
            item_metadata = {
                'title': metadata.get('title', 'Unknown'),
                'url': metadata.get('link', ''),
                'source': metadata.get('source', 'Unknown'),
                'pubDate': metadata.get('pubDate', ''),
                'file_path': file_path
            }
            
            # Additional data for content filtering use case
            additional_data = {
                'content_snippet': content[:300] + '...' if len(content) > 300 else content,
                'cached': is_cached
            }
            
            # Log the evaluation, passing is_cached parameter
            eval_system.log_evaluation(
                use_case='content_filter',
                item_id=item_id,
                item_metadata=item_metadata,
                eval_result=evaluation_result,
                additional_data=additional_data,
                is_cached=is_cached
            )
            
            # Add to filtered list if passes
            if evaluation_result['pass_filter']:
                filtered_articles.append({
                    'file_path': file_path,
                    'metadata': metadata,
                    'evaluation': evaluation_result
                })
                logger.info(f"Article PASSED: {metadata.get('title', 'Unknown')}")
            else:
                logger.info(f"Article FAILED: {metadata.get('title', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Error processing article {file_path}: {e}")
    
    # Log cache stats if used
    if use_cache:
        logger.info(f"Cache hit rate: {cache_hits}/{len(article_files)} ({cache_hits/len(article_files)*100:.1f}%)")
    
    # Limit to max articles
    filtered_articles = filtered_articles[:FILTERING_CONFIG.get('max_articles_per_digest', 10)]
    
    logger.info(f"Selected {len(filtered_articles)} articles out of {len(article_files)} total")
    
    return filtered_articles

# Replace the save_human_feedback function with one that uses the evaluation system
def save_human_feedback(evaluation_id, human_decision, human_notes=None):
    """
    Save human feedback on an evaluation.
    
    Args:
        evaluation_id: ID of the evaluation entry
        human_decision: Boolean pass/fail decision
        human_notes: Optional notes from human reviewer
    """
    eval_system = EvaluationSystem(EVALUATION_CONFIG)
    return eval_system.save_human_feedback(evaluation_id, human_decision, human_notes)

# Remove get_feedback_stats function or replace it with:
def get_feedback_stats(days=30):
    """
    Get statistics on human feedback.
    
    Args:
        days: Number of days to include in stats
        
    Returns:
        dict: Statistics on human feedback
    """
    eval_system = EvaluationSystem(EVALUATION_CONFIG)
    return eval_system.get_agreement_stats(use_case='content_filter', days=days)
                         
                         
def save_human_feedback(evaluation_id, human_decision, human_notes=None):
    """
    Save human feedback on an evaluation.
    
    Args:
        evaluation_id: ID of the evaluation entry
        human_decision: Boolean pass/fail decision
        human_notes: Optional notes from human reviewer
    """
    try:
        # Load the evaluation file
        eval_files = os.listdir(FILTERING_CONFIG.get('evaluation_logs_directory', 'evaluation_logs'))
        for eval_file in sorted(eval_files, reverse=True):  # Start with most recent
            eval_path = os.path.join(
                FILTERING_CONFIG.get('evaluation_logs_directory', 'evaluation_logs'),
                eval_file
            )
            with open(eval_path, 'r', encoding='utf-8') as f:
                evaluations = json.load(f)
            
            # Find and update the evaluation
            for eval_entry in evaluations:
                if eval_entry.get('id') == evaluation_id:
                    eval_entry['human_validated'] = True
                    eval_entry['human_decision'] = human_decision
                    eval_entry['human_notes'] = human_notes
                    eval_entry['human_validation_timestamp'] = datetime.now().isoformat()
                    
                    # Save the updated evaluations
                    with open(eval_path, 'w', encoding='utf-8') as f:
                        json.dump(evaluations, f, indent=2)
                    
                    logger.info(f"Saved human feedback for evaluation {evaluation_id}")
                    return True
        
        logger.error(f"Could not find evaluation with ID {evaluation_id}")
        return False
    
    except Exception as e:
        logger.error(f"Error saving human feedback: {e}")
        return False

def get_feedback_stats(days=30):
    """
    Get statistics on human feedback.
    
    Args:
        days: Number of days to include in stats
        
    Returns:
        dict: Statistics on human feedback
    """
    try:
        stats = {
            'total_evaluations': 0,
            'human_validated': 0,
            'model_human_agreement': 0,
            'model_false_positives': 0,
            'model_false_negatives': 0,
            'agreement_rate': 0.0
        }
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        # Load evaluation files
        eval_dir = FILTERING_CONFIG.get('evaluation_logs_directory', 'evaluation_logs')
        eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.json')]
        
        for eval_file in eval_files:
            eval_path = os.path.join(eval_dir, eval_file)
            with open(eval_path, 'r', encoding='utf-8') as f:
                evaluations = json.load(f)
            
            for eval_entry in evaluations:
                eval_timestamp = eval_entry.get('evaluation_timestamp', '')
                
                # Skip if before cutoff
                if eval_timestamp < cutoff_str:
                    continue
                
                stats['total_evaluations'] += 1
                
                if eval_entry.get('human_validated', False):
                    stats['human_validated'] += 1
                    
                    model_decision = eval_entry.get('evaluation', {}).get('pass_filter', False)
                    human_decision = eval_entry.get('human_decision', False)
                    
                    if model_decision == human_decision:
                        stats['model_human_agreement'] += 1
                    elif model_decision and not human_decision:
                        stats['model_false_positives'] += 1
                    elif not model_decision and human_decision:
                        stats['model_false_negatives'] += 1
        
        # Calculate agreement rate
        if stats['human_validated'] > 0:
            stats['agreement_rate'] = stats['model_human_agreement'] / stats['human_validated']
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        return {}

if __name__ == "__main__":
    # Example standalone usage
    from config import COLLECTION_CONFIG
    articles_dir = COLLECTION_CONFIG['articles_directory']
    selected = filter_articles(articles_dir, use_cache=True)
    print(f"Selected {len(selected)} articles")
    for article in selected:
        print(f"- {article['metadata'].get('title', 'Unknown')} (PASSED)")