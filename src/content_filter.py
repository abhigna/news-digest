import os
import json
import logging
import requests
from datetime import datetime
import time
import hashlib
from config import INTERESTS, FILTERING_CONFIG, OPENROUTER_CONFIG, JUDGE_CONFIG
from judge_system import JudgeSystem
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from llm_gateway import LlmGateway

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

class ContentFilter:
    """
    Content filtering system that evaluates content against user interests.
    Uses LlmGateway for LLM interactions.
    """
    
    def __init__(self, 
                 filter_config: Dict[str, Any], 
                 llm_gateway: LlmGateway,
                 interests: Dict[str, List[str]],
                 judge_system: Optional[Any] = None):
        """
        Initialize the content filter.
        
        Args:
            filter_config: Dictionary containing filter configuration
            llm_gateway: LlmGateway instance for LLM interactions
            interests: Dictionary containing user interests
            judge_system: Optional JudgeSystem instance for feedback examples
        """
        self.filter_config = filter_config
        self.llm_gateway = llm_gateway
        self.interests = interests
        self.judge_system = judge_system
        
        # Set up cache directory
        self.cache_dir = filter_config.get('llm_cache', 'llm_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def evaluate_content(self, article_content: str, article_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate content using LLM with a binary pass/fail approach.
        
        Args:
            article_content: The full content of the article
            article_metadata: Metadata about the article (title, date, etc.)
            
        Returns:
            dict: Contains pass/fail decision, reasoning, and topics
        """
        # Prepare prompt for the LLM with detailed interest descriptions
        primary_interests = self.interests.get("primary_interests", [])
        excluded_interests = self.interests.get("excluded_interests", [])
        
        # Format interests in a more descriptive way
        primary_interests_text = "\n".join([f"- {interest}" for interest in primary_interests])
        excluded_interests_text = "\n".join([f"- {interest}" for interest in excluded_interests])
        
        title = article_metadata.get("title", "")
        source = article_metadata.get("source", "")
        
        # Truncate content if it's too long to fit in LLM context
        max_content_chars = 6000
        truncated_content = article_content[:max_content_chars] + "..." if len(article_content) > max_content_chars else article_content
        
        # Get feedback examples text
        examples_text = ""
        if self.filter_config.get('include_feedback_examples', False):
            examples_text = self._get_feedback_examples()
        
        # Only include examples section if we actually have examples
        examples_section = f"{examples_text}\n" if examples_text else ""
        
        # Construct the prompt
        prompt = f"""
You are a content filtering system for a personalized tech news digest.

USER INTERESTS DESCRIPTION:
The user is specifically interested in:
{primary_interests_text}

The user is NOT interested in:
{excluded_interests_text}
{examples_section}
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

        # Generate item_id from article metadata
        item_id = self._generate_item_id(article_metadata)
        
        # Generate cache key using interests hash
        interests_hash = self._get_interests_hash()
        cache_key = f"{item_id}_{interests_hash}"
        
        try:
            # Request evaluation from LLM gateway
            result = self.llm_gateway.generate_response(
                prompt=prompt,
                response_model=ContentEvaluation,
                item_id=item_id,
                item_metadata=article_metadata,
                additional_data={
                    'content_snippet': article_content[:300] + '...' if len(article_content) > 300 else article_content
                },
                cache_key=cache_key,
                use_cache=self.filter_config.get('use_cache', True)
            )
            
            # Convert result to dict
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return result
            
        except Exception as e:
            logger.error(f"Error in content evaluation: {e}")
            return {
                'pass_filter': False,
                'main_topics': [],
                'reasoning': f"Error processing article: {str(e)}",
                'specific_interests_matched': []
            }
    
    def filter_articles(self, articles_directory: str, use_cache: bool = True, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter articles based on binary pass/fail evaluation.
        
        Args:
            articles_directory: Directory containing article JSON files
            use_cache: Whether to use cached evaluations when available
            days: Only process articles from the last X days (None means all articles)
            
        Returns:
            list: filtered_articles containing articles that passed the filter
        """
        # Get all article files
        article_files = [f for f in os.listdir(articles_directory) if f.endswith('.json')]
        logger.info(f"Found {len(article_files)} articles to filter")
        
        filtered_articles = []
        
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
                
                # Add file path to metadata for reference
                metadata['file_path'] = file_path
                
                # Evaluate content
                logger.info(f"Evaluating article: {metadata.get('title', 'Unknown')}")
                evaluation_result = self.evaluate_content(content, metadata)
                
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
        
        # Limit to max articles
        filtered_articles = filtered_articles[:self.filter_config.get('max_articles_per_digest', 10)]
        
        logger.info(f"Selected {len(filtered_articles)} articles out of {len(article_files)} total")
        
        return filtered_articles
    
    def _get_interests_hash(self) -> str:
        """
        Generate a hash representing the current interests configuration.
        This allows us to detect changes in the interests.
        
        Returns:
            str: Hash of the current interests configuration
        """
        # Convert interests to a string representation for hashing
        interests_str = json.dumps(self.interests, sort_keys=True)
        return hashlib.md5(interests_str.encode()).hexdigest()
    
    def _generate_item_id(self, article_metadata: Dict[str, Any]) -> str:
        """
        Generate a unique ID for an article based on metadata.
        
        Args:
            article_metadata: Metadata about the article
            
        Returns:
            str: Unique ID for the article
        """
        # Generate ID from title and source if available
        title = article_metadata.get('title', '')
        source = article_metadata.get('source', '')
        
        if title and source:
            return hashlib.md5(f"{title}:{source}".encode()).hexdigest()
        
        # Fall back to file path if available
        file_path = article_metadata.get('file_path', '')
        if file_path:
            return os.path.basename(file_path).split('.')[0]
        
        # Last resort: generate random ID
        return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
    
    def _get_feedback_examples(self) -> str:
        """
        Get feedback examples for the prompt.
        
        Returns:
            str: Formatted feedback examples text
        """
        if not self.judge_system:
            return ""
        
        # Get examples from the judge system
        examples_count = self.filter_config.get("examples_count", 2)
        return self.judge_system.get_feedback_examples(
            use_case='content_filter',
            count=examples_count
        )
