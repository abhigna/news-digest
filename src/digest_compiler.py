import os
import json
import logging
from datetime import datetime
import time
from collections import defaultdict
import math
from config import DIGEST_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tech_digest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("digest_compiler")

def group_articles_by_topic(articles):
    """
    Group articles into topics based on their main topics.
    
    Args:
        articles: List of articles with metadata, relevance, and summary information
        
    Returns:
        dict: Articles grouped by topic
    """
    # Initialize topic groups
    topic_groups = defaultdict(list)
    
    # For each article, find its main topic
    for article in articles:
        # Get main topics from relevance info
        main_topics = article.get('relevance', {}).get('main_topics', [])
        
        if not main_topics:
            # If no topics, put in "Miscellaneous"
            topic_groups["Miscellaneous"].append(article)
            continue
        
        # Use the first/main topic for grouping
        primary_topic = main_topics[0]
        topic_groups[primary_topic].append(article)
    
    # Convert defaultdict to regular dict
    return dict(topic_groups)

def format_digest(articles, group_by_topics=True):
    """
    Format articles into a readable digest.
    
    Args:
        articles: List of articles with metadata, relevance, and summary information
        group_by_topics: Whether to group articles by topic
        
    Returns:
        str: Formatted digest text
    """
    # Get current date for title
    current_date = datetime.now().strftime("%Y-%m-%d")
    digest_title = DIGEST_CONFIG['digest_title_format'].format(date=current_date)
    
    # Start with the title
    digest_text = f"# {digest_title}\n\n"
    
    if group_by_topics and DIGEST_CONFIG['group_by_topics']:
        # Group articles by topic
        topic_groups = group_articles_by_topic(articles)
        
        # Limit number of topic clusters if needed
        if len(topic_groups) > DIGEST_CONFIG['max_topic_clusters']:
            # Keep only the topics with the most articles
            sorted_topics = sorted(topic_groups.items(), key=lambda x: len(x[1]), reverse=True)
            limited_topics = sorted_topics[:DIGEST_CONFIG['max_topic_clusters']]
            
            # Make a new dictionary with just those topics
            topic_groups = {topic: articles for topic, articles in limited_topics}
        
        # Add a table of contents
        digest_text += "## Topics\n\n"
        for topic in topic_groups.keys():
            digest_text += f"- [{topic}](#{topic.lower().replace(' ', '-')})\n"
        digest_text += "\n"
        
        # Add each topic section
        for topic, topic_articles in topic_groups.items():
            digest_text += f"## {topic}\n\n"
            
            # Add each article in this topic
            for article in topic_articles:
                digest_text += format_article(article)
    else:
        # Just list all articles without grouping
        for article in articles:
            digest_text += format_article(article)
    
    # Add footer
    digest_text += "\n---\n"
    digest_text += f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return digest_text

def format_article(article):
    """Format a single article for the digest."""
    metadata = article.get('metadata', {})
    summary = article.get('summary', {})
    
    title = metadata.get('title', 'Unknown Title')
    url = metadata.get('link', '')
    author = metadata.get('creator', 'Unknown Author')
    pub_date = metadata.get('pubDate', '')
    
    # Format the date nicely if possible
    try:
        for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
            try:
                parsed_date = datetime.strptime(pub_date, fmt)
                pub_date = parsed_date.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue
    except Exception:
        # Keep the original if parsing fails
        pass
    
    # Get summary elements
    summary_text = summary.get('summary', 'No summary available.')
    key_points = summary.get('key_points', [])
    technical_details = summary.get('technical_details', [])
    
    # Start building the article section
    article_text = f"### [{title}]({url})\n\n"
    article_text += f"*By {author} | {pub_date}*\n\n"
    article_text += f"{summary_text}\n\n"
    
    # Add key points if available
    if key_points:
        article_text += "**Key Points:**\n"
        for point in key_points:
            article_text += f"- {point}\n"
        article_text += "\n"
    
    # Add technical details if available
    if technical_details:
        article_text += "**Technical Details:** "
        article_text += ", ".join(technical_details)
        article_text += "\n\n"
    
    # Add a separator
    article_text += "---\n\n"
    
    return article_text

def compile_digest(summarized_articles):
    """
    Compile the final digest from summarized articles.
    
    Args:
        summarized_articles: List of articles with summaries
        
    Returns:
        str: Path to the generated digest file
    """
    # Format the digest content
    digest_content = format_digest(summarized_articles)
    
    # Create output filename
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_filename = DIGEST_CONFIG['output_file_format'].format(date=current_date)
    
    # Write to file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(digest_content)
        
        logger.info(f"Digest compiled successfully: {output_filename}")
        return output_filename
    
    except Exception as e:
        logger.error(f"Error writing digest file: {e}")
        return None

def print_digest_to_console(digest_file):
    """Print the digest to the console."""
    try:
        with open(digest_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("\n" + "="*80)
        print(content)
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error reading digest file: {e}")

if __name__ == "__main__":
    # Example standalone usage
    from content_filter import filter_articles
    from content_summarizer import summarize_selected_articles
    from config import COLLECTION_CONFIG
    
    articles_dir = COLLECTION_CONFIG['articles_directory']
    selected = filter_articles(articles_dir)
    summarized = summarize_selected_articles(selected)
    
    digest_file = compile_digest(summarized)
    if digest_file:
        print_digest_to_console(digest_file)