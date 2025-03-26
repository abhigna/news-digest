import json
import logging
import requests
import time
from config import SUMMARIZATION_CONFIG, OPENROUTER_CONFIG
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from models import ContentSummaryModelResponse

# Set up logging
logger = logging.getLogger(__name__)

# Define the structured output model
class ArticleSummary(BaseModel):
    summary: str = Field(description="A concise summary of the article with the main subject italicized")
    key_points: List[str] = Field(description="3-5 key points from the article, formatted as a list", default=[])

class ContentSummarizer:
    """
    Content summarization system that generates summaries for articles using LLM.
    Uses LlmGateway for LLM interactions.
    """
    
    def __init__(self, 
                 summarization_config: Dict[str, Any], 
                 llm_gateway,
                 judge_system=None):
        """
        Initialize the content summarizer.
        
        Args:
            summarization_config: Dictionary containing summarization configuration
            llm_gateway: LlmGateway instance for LLM interactions
            judge_system: Optional JudgeSystem instance for feedback examples
        """
        self.config = summarization_config
        self.llm_gateway = llm_gateway
        self.judge_system = judge_system
    
    def summarize_article(self, article_content, article_metadata, evaluation_info):
        """
        Generate a concise summary of an article using LLM.
        
        Args:
            article_content: The full content of the article
            article_metadata: Metadata about the article (title, date, etc.)
            evaluation_info: Information about article evaluation (topics, pass/fail)
            
        Returns:
            ContentSummaryModelResponse: Contains the summary and key points
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
        
        max_length = self.config["max_summary_length"]
        
        # Get feedback examples text
        examples_text = ""
        if self.config.get('include_feedback_examples', False) and self.judge_system:
            examples_text = self.judge_system.get_feedback_examples(
                use_case='content_summarizer',
                count=self.config.get("examples_count", 2)
            )
        
        # Only include examples section if we actually have examples
        examples_section = f"{examples_text}\n" if examples_text else ""
        
        prompt = f"""
You are a technical content summarizer for a tech news digest aimed at software engineers and developers.

ARTICLE TO SUMMARIZE:
Title: {title}
URL: {url}
Author: {author}
Topics: {topics_str}
Content: {truncated_content}
{examples_section}
TASK:
Create a comprehensive technical summary of this article with two components:

1. Summary: Write a direct, crisp {max_length}-word summary that captures the main technological points and significance to developers. Focus on technical details, tools, frameworks, or concepts mentioned.

2. Key Points: Extract 3-5 key points or takeaways from the article that would be most relevant to developers.

IMPORTANT GUIDELINES:
1. Start immediately with the key information - avoid phrases like "The article explores", "This article discusses", etc.
2. Use active voice and direct language throughout.
3. Italicize the main subject/sentence by surrounding it with asterisks (*like this*).
4. Focus on what developers need to know, not on describing the article itself.
5. Be specific about technologies, tools, methods, or frameworks mentioned.
"""

        # Generate item_id from article metadata
        item_id = self._generate_item_id(article_metadata)
        
        try:
            # Request summary from LLM gateway
            result = self.llm_gateway.generate_response(
                prompt=prompt,
                response_model=ArticleSummary,
                item_id=item_id,
                item_metadata=article_metadata,
                additional_data={
                    'content_snippet': article_content[:300] + '...' if len(article_content) > 300 else article_content
                },
                cache_key=item_id,
                use_cache=self.config.get('use_cache', True)
            )
            
            # Convert to our standardized response model
            response = ContentSummaryModelResponse(
                id=item_id,
                item_metadata=article_metadata,
                is_cached=result.get('is_cached', False) if isinstance(result, dict) else False,
                summary=result.summary if hasattr(result, 'summary') else result.get('summary', ''),
                key_points=result.key_points if hasattr(result, 'key_points') else result.get('key_points', [])
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in article summarization: {e}")
            return ContentSummaryModelResponse(
                id=item_id,
                item_metadata=article_metadata,
                summary=f"Error: {str(e)}",
                key_points=[]
            )
    
    def summarize_selected_articles(self, selected_articles):
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
                summary_result = self.summarize_article(content, metadata, evaluation)
                
                # Add summary to article data
                article_with_summary = article.copy()
                article_with_summary['summary'] = summary_result
                
                summarized_articles.append(article_with_summary)
                
            except Exception as e:
                logger.error(f"Error summarizing article {file_path}: {e}")
        
        logger.info(f"Summarized {len(summarized_articles)} articles")
        
        return summarized_articles
    
    def _generate_item_id(self, article_metadata):
        """
        Generate a unique ID for an article based on metadata.
        
        Args:
            article_metadata: Metadata about the article
            
        Returns:
            str: Unique ID for the article
        """
        import hashlib
        
        # Generate ID from title and source if available
        title = article_metadata.get('title', '')
        source = article_metadata.get('source', '')
        
        if title and source:
            return hashlib.md5(f"{title}:{source}".encode()).hexdigest()
        
        # Fall back to file path if available
        file_path = article_metadata.get('file_path', '')
        if file_path:
            import os
            return os.path.basename(file_path).split('.')[0]
        
        # Last resort: generate random ID
        from datetime import datetime
        return hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()
