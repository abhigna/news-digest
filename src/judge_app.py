import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import uuid
import importlib
import time
import logging
from config import JUDGE_CONFIG, COLLECTION_CONFIG, INTERESTS, CRITIQUE_CONFIG
import requests
from content_filter import ContentEvaluation  
from content_summarizer import ArticleSummary


# Set up logging
logger = logging.getLogger(__name__)

# Constants
ARTICLES_DIR = COLLECTION_CONFIG.get("articles_directory", "articles")
FEEDBACK_FILE = "human_feedback.json"
LLM_TRACE_LOGS_DIR = JUDGE_CONFIG.get('llm_trace_logs')

# Ensure directories exist
os.makedirs(ARTICLES_DIR, exist_ok=True)
os.makedirs(LLM_TRACE_LOGS_DIR, exist_ok=True)

# Initialize NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class JudgeApp:
    """A system for managing content judgments and decisions."""
    
    def __init__(self, config):
        self.config = config
        self.llm_trace_logs_dir = config.get('llm_trace_logs')
        os.makedirs(self.llm_trace_logs_dir, exist_ok=True)
    
    def get_evaluations(self, use_case=None, days=30, validated_only=False):
        """Get evaluations with filtering options."""
        filtered_evals = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get evaluation files
        if not os.path.exists(self.llm_trace_logs_dir):
            return []
            
        eval_files = [f for f in os.listdir(self.llm_trace_logs_dir) if f.endswith('.json') and f != "article_exceptions.json" and f != FEEDBACK_FILE]
        
        for eval_file in eval_files:
            file_path = os.path.join(self.llm_trace_logs_dir, eval_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
                
                for eval_entry in evaluations:
                    # Normalize evaluation data structure
                    eval_entry = self.normalize_evaluation(eval_entry)
                    
                    # Apply filters
                    if use_case and eval_entry.get('use_case') != use_case:
                        continue
                    
                    if validated_only and not eval_entry.get('human_validated', False):
                        continue
                    
                    # Apply date filter
                    timestamp_str = eval_entry.get('evaluation_timestamp', '')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str)
                            if timestamp < cutoff_date:
                                continue
                        except ValueError:
                            pass
                    
                    filtered_evals.append(eval_entry)
                    
            except Exception as e:
                logger.error(f"Error reading evaluation file {eval_file}: {e}")
        
        return filtered_evals
    
    def save_human_feedback(self, eval_id, human_decision, human_notes=None):
        """Save human feedback for an evaluation."""
        try:
            updated = False
            
            # Find the evaluation in log files
            eval_files = [f for f in os.listdir(self.llm_trace_logs_dir) if f.endswith('.json') and f != "article_exceptions.json" and f != FEEDBACK_FILE]
            
            for eval_file in eval_files:
                file_path = os.path.join(self.llm_trace_logs_dir, eval_file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        evaluations = json.load(f)
                    
                    # Look for the evaluation entry
                    for eval_entry in evaluations:
                        if eval_entry.get('id') == eval_id:
                            # Update with human feedback
                            eval_entry['human_validated'] = True
                            eval_entry['human_decision'] = human_decision
                            eval_entry['human_notes'] = human_notes
                            eval_entry['human_validation_timestamp'] = datetime.now().isoformat()
                            updated = True
                            break
                    
                    # If found and updated, save back to file
                    if updated:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(evaluations, f, indent=2)
                        
                        # Also save to feedback file for easy reference
                        self._add_to_feedback_file(eval_id, human_decision, human_notes)
                        
                        logger.info(f"Saved human feedback for evaluation {eval_id}")
                        return True
                
                except Exception as e:
                    logger.error(f"Error processing evaluation file {eval_file}: {e}")
            
            if not updated:
                logger.error(f"Could not find evaluation with ID {eval_id}")
            
            return updated
            
        except Exception as e:
            logger.error(f"Error saving human feedback: {e}")
            return False
    
    def get_agreement_stats(self, use_case=None, days=30):
        """Calculate agreement statistics."""
        # Get evaluations
        evaluations = self.get_evaluations(use_case=use_case, days=days)
        
        # Calculate stats
        stats = {
            'total': len(evaluations),
            'validated': 0,
            'agreement': 0,
            'agreement_rate': 0,
            'false_positives': 0,
            'false_negatives': 0
        }
        
        # Count validated evaluations
        validated_evals = [e for e in evaluations if e.get('human_validated', False)]
        stats['validated'] = len(validated_evals)
        
        if stats['validated'] == 0:
            return stats
        
        # Calculate agreement metrics
        for eval_entry in validated_evals:
            # Get the judge decision - handle different formats
            judge_decision = eval_entry.get('judge_decision', {})
            
            # Get decisions
            
            if isinstance(judge_decision, ContentEvaluation):
                model_decision = judge_decision.pass_filter
            else:
                model_decision = judge_decision.get('pass_filter')
            human_decision = eval_entry.get('human_decision', False)
            
            if model_decision == human_decision:
                stats['agreement'] += 1
            elif model_decision and not human_decision:
                stats['false_positives'] += 1
            elif not model_decision and human_decision:
                stats['false_negatives'] += 1
        
        # Calculate agreement rate
        if stats['validated'] > 0:
            stats['agreement_rate'] = stats['agreement'] / stats['validated']
        
        return stats
    
    def normalize_evaluation(self, eval_entry):
        """Normalize evaluation data structure to handle different formats."""
        normalized = eval_entry.copy()
        
        # Ensure standard fields exist
        if 'file_path' not in normalized:
            file_path = normalized.get('item_metadata', {}).get('file_path', '')
            if not file_path:
                # Try to find from other fields
                item_id = normalized.get('item_id', '')
                if item_id:
                    possible_path = os.path.join(ARTICLES_DIR, f"{item_id}.json")
                    if os.path.exists(possible_path):
                        file_path = possible_path
            normalized['file_path'] = file_path
            
        if 'title' not in normalized:
            title = normalized.get('item_metadata', {}).get('title', 'Unknown')
            normalized['title'] = title
            
        # Process different use cases differently
        use_case = normalized.get('use_case', 'content_filter')
        
        if use_case == "content_filter":
            # Get the response data which contains the ContentEvaluation
            response_data = normalized.get('response', {})
            if response_data:
                try:
                    # Try to convert to ContentEvaluation if it's not already
                    if not isinstance(response_data, ContentEvaluation):
                        # If it's a dict, convert to ContentEvaluation
                        judge_decision = ContentEvaluation(
                            pass_filter=response_data.get('pass_filter', False),
                            main_topics=response_data.get('main_topics', []),
                            reasoning=response_data.get('reasoning', ''),
                            specific_interests_matched=response_data.get('specific_interests_matched', [])
                        )
                    else:
                        judge_decision = response_data
                    normalized['judge_decision'] = judge_decision
                except Exception as e:
                    logger.error(f"Error parsing ContentEvaluation: {e}")
                    normalized['judge_decision'] = {
                        'pass_filter': False,
                        'reasoning': 'Error parsing evaluation data',
                        'main_topics': [],
                        'specific_interests_matched': []
                    }
            elif 'judge_decision' not in normalized:
                normalized['judge_decision'] = {
                    'pass_filter': False,
                    'reasoning': 'No evaluation data found',
                    'main_topics': [],
                    'specific_interests_matched': []
                }
        
        elif use_case == "content_summarizer":
            # For content_summarizer, the structure is different
            response_data = normalized.get('response', {})
            if response_data:
                try:
                    # Try to convert to ArticleSummary if it's not already
                    if not isinstance(response_data, ArticleSummary):
                        # If it's a dict, convert to ArticleSummary
                        judge_decision = ArticleSummary(
                            summary=response_data.get('summary', ''),
                            key_points=response_data.get('key_points', []),
                            technical_details=response_data.get('technical_details', [])
                        )
                    else:
                        judge_decision = response_data
                    normalized['judge_decision'] = judge_decision
                except Exception as e:
                    logger.error(f"Error parsing ArticleSummary: {e}")
                    normalized['judge_decision'] = {
                        'summary': 'Error parsing summary data',
                        'key_points': [],
                        'technical_details': []
                    }
            elif 'judge_decision' not in normalized:
                normalized['judge_decision'] = {
                    'summary': 'No summary data found',
                    'key_points': [],
                    'technical_details': []
                }
        
        return normalized
    
    def _add_to_feedback_file(self, eval_id, human_decision, human_notes):
        """Add feedback to a separate feedback file for easier analysis."""
        feedback_path = os.path.join(self.llm_trace_logs_dir, FEEDBACK_FILE)
        
        # Load existing feedback or create new
        if os.path.exists(feedback_path):
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        else:
            feedback_data = []
        
        # Find the evaluation in log files to get file path
        file_path = None
        title = None
        content = None
        
        eval_files = [f for f in os.listdir(self.llm_trace_logs_dir) if f.endswith('.json') and f != "article_exceptions.json" and f != FEEDBACK_FILE]
        
        for eval_file in eval_files:
            file_path_full = os.path.join(self.llm_trace_logs_dir, eval_file)
            
            try:
                with open(file_path_full, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
                
                # Look for the evaluation entry
                for eval_entry in evaluations:
                    if eval_entry.get('id') == eval_id:
                        file_path = eval_entry.get('file_path', '')
                        title = eval_entry.get('title', 'Unknown')
                        break
                
                if file_path:
                    break
                
            except Exception as e:
                logger.error(f"Error reading evaluation file {eval_file}: {e}")
        
        # Get article content if file path is available
        if file_path:
            content_text, _ = get_article_content(file_path)
            content = content_text
        
        # Add new feedback with content
        feedback_entry = {
            'eval_id': eval_id,
            'human_decision': human_decision,
            'human_notes': human_notes,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add content and title if available
        if content:
            feedback_entry['content'] = content
        if title:
            feedback_entry['title'] = title
        if file_path:
            feedback_entry['file_path'] = file_path
        
        feedback_data.append(feedback_entry)
        
        # Save updated feedback
        with open(feedback_path, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, indent=2)


def get_article_content(file_path):
    """Get the content of an article with improved error handling."""
    try:
        # Try different ways to locate the file
        found_path = None
        
        # 1. Check if the file exists as given
        if os.path.exists(file_path):
            found_path = file_path
        
        # 2. Check if the file exists in the articles directory
        elif not os.path.isabs(file_path):
            base_name = os.path.basename(file_path)
            possible_path = os.path.join(ARTICLES_DIR, base_name)
            if os.path.exists(possible_path):
                found_path = possible_path
        
        # 3. Try to extract ID from filename and construct path
        if not found_path:
            try:
                file_id = os.path.basename(file_path).split('.')[0]
                possible_path = os.path.join(ARTICLES_DIR, f"{file_id}.json")
                if os.path.exists(possible_path):
                    found_path = possible_path
            except:
                pass
        
        if not found_path:
            logger.error(f"Article file not found: {file_path}")
            return "", {}
            
        with open(found_path, 'r', encoding='utf-8') as f:
            article_data = json.load(f)
            
        content = article_data.get('content', '')
        metadata = article_data.get('metadata', {})
        
        return content, metadata
    except Exception as e:
        logger.error(f"Error loading article {file_path}: {e}")
        return "", {}


def get_article_exceptions():
    """Get all article exceptions."""
    try:
        exceptions_file = os.path.join(LLM_TRACE_LOGS_DIR, "article_exceptions.json")
        
        if os.path.exists(exceptions_file):
            with open(exceptions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        st.error(f"Error loading article exceptions: {e}")
        return []


def save_article_exception(exception_data):
    """Save an article exception."""
    try:
        exceptions_file = os.path.join(LLM_TRACE_LOGS_DIR, "article_exceptions.json")
        
        # Load existing exceptions
        if os.path.exists(exceptions_file):
            with open(exceptions_file, 'r', encoding='utf-8') as f:
                exceptions = json.load(f)
        else:
            exceptions = []
        
        # Add new exception with timestamp
        exception_data['timestamp'] = datetime.now().isoformat()
        exceptions.append(exception_data)
        
        # Save updated exceptions
        with open(exceptions_file, 'w', encoding='utf-8') as f:
            json.dump(exceptions, f, indent=2)
        
        return True
    except Exception as e:
        st.error(f"Error saving article exception: {e}")
        return False


def remove_exception(article_id):
    """Remove an article exception."""
    try:
        exceptions_file = os.path.join(LLM_TRACE_LOGS_DIR, "article_exceptions.json")
        
        if os.path.exists(exceptions_file):
            with open(exceptions_file, 'r', encoding='utf-8') as f:
                exceptions = json.load(f)
            
            # Filter out the exception to remove
            updated_exceptions = [e for e in exceptions if e.get('article_id') != article_id]
            
            # Save updated exceptions
            with open(exceptions_file, 'w', encoding='utf-8') as f:
                json.dump(updated_exceptions, f, indent=2)
            
            return True
    except Exception as e:
        st.error(f"Error removing article exception: {e}")
        return False


def save_config_changes(interests_to_add=None, interests_to_remove=None):
    """Save changes to interests configuration."""
    try:
        # Load current config file content
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        # Process additions to primary interests
        if interests_to_add:
            # Find where primary_interests list ends
            primary_interests_start = config_content.find('"primary_interests": [')
            if primary_interests_start > -1:
                # Find the closing bracket
                bracket_end = config_content.find(']', primary_interests_start)
                
                # Insert new interests before the closing bracket
                new_interests = ',\n        '.join([f'"{interest}"' for interest in interests_to_add])
                if config_content[bracket_end-1] != '[':  # If list is not empty
                    new_interests = ',\n        ' + new_interests
                
                config_content = config_content[:bracket_end] + new_interests + config_content[bracket_end:]
        
        # Process removals from excluded interests
        if interests_to_remove:
            # Add to excluded interests
            excluded_interests_start = config_content.find('"excluded_interests": [')
            if excluded_interests_start > -1:
                # Find the closing bracket
                bracket_end = config_content.find(']', excluded_interests_start)
                
                # Insert new exclusions before the closing bracket
                new_exclusions = ',\n        '.join([f'"{interest}"' for interest in interests_to_remove])
                if config_content[bracket_end-1] != '[':  # If list is not empty
                    new_exclusions = ',\n        ' + new_exclusions
                
                config_content = config_content[:bracket_end] + new_exclusions + config_content[bracket_end:]
        
        # Save updated config
        with open('config.py', 'w') as f:
            f.write(config_content)
        
        # Reload the config module
        import config
        importlib.reload(config)
        
        return True
    except Exception as e:
        st.error(f"Failed to update config: {e}")
        return False


def extract_topics(evaluations):
    """Extract and count topics from evaluations."""
    topics_counter = Counter()
    
    for eval_entry in evaluations:
        judge_decision = eval_entry.get('judge_decision', {})
        
        # Handle both Pydantic model and dict formats
        if isinstance(judge_decision, ContentEvaluation):
            topics = judge_decision.main_topics
        elif isinstance(judge_decision, ArticleSummary):
            topics = judge_decision.key_points
            
        topics_counter.update(topics)
    
    return topics_counter


def extract_interest_matches(evaluations):
    """Extract and count interests matched from evaluations."""
    interest_counter = Counter()
    
    for eval_entry in evaluations:
        judge_decision = eval_entry.get('judge_decision', {})
        
        # Handle both Pydantic model and dict formats
        if isinstance(judge_decision, ContentEvaluation):
            interests = judge_decision.specific_interests_matched
        elif isinstance(judge_decision, ArticleSummary):
            interests = judge_decision.technical_details
            
        interest_counter.update(interests)
    
    return interest_counter


def generate_word_cloud(evaluations):
    """Generate word count data from evaluation reasoning."""
    stop_words = set(stopwords.words('english'))
    words = []
    
    for eval_entry in evaluations:
        judge_decision = eval_entry.get('judge_decision', {})
        
        # Handle both Pydantic model and dict formats
        if isinstance(judge_decision, ContentEvaluation):
            reasoning = judge_decision.reasoning
        else:
            reasoning = judge_decision.get('reasoning', '')
            
        if reasoning:
            tokens = word_tokenize(reasoning.lower())
            words.extend([word for word in tokens if word.isalpha() and word not in stop_words])
    
    return Counter(words)


def generate_model_critique(eval_entry, original_context=None):
    """Generate a critique of the model's decision for an evaluation."""
    try:
        # Extract evaluation data
        judge_decision = eval_entry.get('judge_decision', {})
        
        # Handle both Pydantic model and dict formats for content_filter
        if isinstance(judge_decision, ContentEvaluation):
            model_decision = judge_decision.pass_filter
            reasoning = judge_decision.reasoning
            topics = judge_decision.main_topics
            interests = judge_decision.specific_interests_matched
        # Handle both Pydantic model and dict formats for content_summarizer
        elif isinstance(judge_decision, ArticleSummary):
            model_decision = True  # Summarizers don't have pass/fail
            reasoning = judge_decision.summary
            topics = []
            interests = []
        # Fallback to dictionary access
        else:
            model_decision = judge_decision.get('pass_filter', False)
            reasoning = judge_decision.get('reasoning', 'No reasoning provided')
            topics = judge_decision.get('main_topics', [])
            interests = judge_decision.get('specific_interests_matched', [])
        
        # Get content and title for the article
        title = eval_entry.get('title', 'Unknown Article')
        content = eval_entry.get('content', '')
        
        # Get content if not directly available in the evaluation
        if not content:
            file_path = eval_entry.get('file_path', '')
            content, _ = get_article_content(file_path)
        
        # Check if required configuration is available
        if not CRITIQUE_CONFIG.get("api_key"):
            logger.warning("No API key found in CRITIQUE_CONFIG, falling back to simple critique")
            return 'error: no api key or original context'
        
        from openai import OpenAI
        
        # Initialize OpenAI client with OpenRouter configuration
        client = OpenAI(
            base_url=CRITIQUE_CONFIG['api_base'],
            api_key=CRITIQUE_CONFIG['api_key']
        )
        
        # Create different system and user prompts based on use case
        selected_use_case = eval_entry.get('use_case', 'content_filter')
        
        if selected_use_case == "content_filter":
            # Format the user interests for the prompt
            primary_interests = INTERESTS.get("primary_interests", [])
            excluded_interests = INTERESTS.get("excluded_interests", [])
            
            primary_interests_formatted = "\n".join([f"- {interest}" for interest in primary_interests])
            excluded_interests_formatted = "\n".join([f"- {interest}" for interest in excluded_interests])
            
            system_prompt = f"""You are an expert AI evaluator. Your task is to critique the decision made by a content filtering model.
            
You will analyze whether the model correctly determined if an article matches the user's interests based on:
1. The content and topic of the article
2. The user's specified interests and exclusions
3. The model's reasoning

Provide a balanced critique addressing:
- Whether the model's decision seems appropriate
- Any interests or exclusions that were overlooked
- Accuracy of the topic analysis
- Overall quality of the reasoning

=== USER INTERESTS ===
Primary Interests:
{primary_interests_formatted}

Excluded Interests:
{excluded_interests_formatted}

=== MODEL EVALUATION ===
Decision: {"PASS" if model_decision else "FAIL"}
Reasoning: {reasoning}
Topics Identified: {', '.join(topics) if topics else 'None'}
Interests Matched: {', '.join(interests) if interests else 'None'}

Return a concise paragraph with your critique, not a structured response."""

            # Create user prompt with only the title and content
            user_prompt = f"""
Please provide a brief critique of whether this content filtering decision is correct based on the user's interests.

=== ARTICLE TITLE ===
{title}

=== ARTICLE CONTENT ===
{content}
"""
            
        elif selected_use_case == "content_summarizer":
            system_prompt = """You are an expert AI evaluator. Your task is to critique the summary generated by a content summarization model.
            
You will analyze whether the model effectively summarized the article by evaluating:
1. Accuracy - Does the summary contain factual errors or misrepresentations?
2. Completeness - Does the summary include the key information from the article?
3. Conciseness - Is the summary appropriately brief without unnecessary details?
4. Clarity - Is the summary well-written and easy to understand?

Return a concise paragraph with your critique, not a structured response."""

            # For summarizer, we need the summary and key points
            summary = judge_decision.get('summary', 'No summary provided')
            key_points = judge_decision.get('key_points', [])
            key_points_formatted = "\n".join([f"- {point}" for point in key_points]) if key_points else "None provided"
            
            user_prompt = f"""
=== ARTICLE TITLE ===
{title}

=== ARTICLE CONTENT ===
{content}

=== MODEL GENERATED SUMMARY ===
{summary}

=== KEY POINTS EXTRACTED ===
{key_points_formatted}

Please provide a brief critique of this summary's quality, accuracy, and completeness.
"""
        
        else:
            # Fallback for any other use case
            system_prompt = "You are an expert AI evaluator. Provide a brief critique of the model's output."
            user_prompt = f"Content: {content}\n\nModel output: {reasoning}\n\nProvide a brief critique."
        
        try:
            # Get simple text output instead of structured output
            response = client.chat.completions.create(
                model=CRITIQUE_CONFIG["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=CRITIQUE_CONFIG["max_tokens"],
                temperature=CRITIQUE_CONFIG["temperature"],
                timeout=CRITIQUE_CONFIG['timeout_seconds']
            )
            
            # Return the simple text critique
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in critique generation: {e}")
            return f"Error generating critique: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error generating model critique: {e}")
        return f"Error generating critique: {str(e)}"


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="Judge App", layout="wide")
    
    st.title("Judge App")
    
    # Initialize evaluation system
    judge_app = JudgeApp(JUDGE_CONFIG)
    
    # Sidebar with use case selection
    st.sidebar.header("Options")
    
    # Update to include content summarizer
    use_cases = ["content_filter", "content_summarizer"]
    selected_use_case = st.sidebar.selectbox("Judge Use Case", use_cases)
    
    # Other sidebar options
    days_back = st.sidebar.slider("Days to include", 1, 90, JUDGE_CONFIG.get("default_days_to_display", 30))
    show_validated = st.sidebar.checkbox("Show already validated items", False)
    
    # Number of items to show per page
    items_per_page = st.sidebar.slider("Items per page", 5, 50, 10)
    
    # Filter options for content filter
    filter_mode = st.sidebar.radio("Filter by", ["All", "Passed filter", "Failed filter"])
    
    # Load evaluations
    evaluations = judge_app.get_evaluations(use_case=selected_use_case, days=days_back)
    
    # Apply filters
    if filter_mode == "Passed filter":
        evaluations = [e for e in evaluations if e.get('judge_decision', {}).get('pass_filter', False)]
    elif filter_mode == "Failed filter":
        evaluations = [e for e in evaluations if not e.get('judge_decision', {}).get('pass_filter', False)]
    
    if not show_validated:
        evaluations = [e for e in evaluations if not e.get('human_validated', False)]
    
    # Create tabs
    tabs = st.tabs(["Evaluate", "Statistics"])
    
    # Tab 1: Evaluate (formerly Batch Evaluate)
    with tabs[0]:
        if not evaluations:
            st.info("No articles found matching the current filters.")
        else:
            st.write(f"Found {len(evaluations)} articles to evaluate. Showing up to {items_per_page} items.")
            
            # Create evaluation data for batch display
            batch_evals = evaluations[:items_per_page]
            
            for i, eval_entry in enumerate(batch_evals):
                # Normalize evaluation data before displaying
                eval_entry = judge_app.normalize_evaluation(eval_entry)
                
                with st.expander(f"{i+1}. {eval_entry.get('title', 'Unknown Article')}", expanded=True):
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Article content
                        file_path = eval_entry.get('file_path', '')
                        content, metadata = get_article_content(file_path)
                        
                        st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
                        st.write(f"**Date:** {metadata.get('pubDate', 'Unknown')}")
                        
                        # Display truncated content with a "Show more" toggle instead of nested expander
                        truncated = len(content) > 1000
                        display_content = content[:1000] + "..." if truncated else content
                        st.markdown(display_content)
                        
                        if truncated:
                            if st.checkbox(f"Show full content for article {i+1}", key=f"full_{eval_entry.get('id')}"):
                                st.markdown(content)
                    
                    with col2:
                        # Model evaluation display
                        judge_decision = eval_entry.get('judge_decision', {})
                        
                        if selected_use_case == "content_filter":
                            # First check if it's a ContentEvaluation object
                            if isinstance(judge_decision, ContentEvaluation):
                                model_decision = "✅ PASS" if judge_decision.pass_filter else "❌ FAIL"
                                topics = judge_decision.main_topics
                                interests = judge_decision.specific_interests_matched
                                reasoning = judge_decision.reasoning
                            else:
                                # Fallback to dictionary access for backward compatibility
                                model_decision = "✅ PASS" if judge_decision.get('pass_filter', False) else "❌ FAIL"
                                topics = judge_decision.get('main_topics', [])
                                interests = judge_decision.get('specific_interests_matched', [])
                                reasoning = judge_decision.get('reasoning', 'No reasoning provided')
                            
                            st.write(f"**Model Decision:** {model_decision}")
                            
                            st.write("**Topics:**")
                            if topics:
                                for topic in topics:
                                    st.write(f"- {topic}")
                            else:
                                st.write("No topics identified")
                            
                            st.write("**Interests Matched:**")
                            if interests:
                                for interest in interests:
                                    st.write(f"- {interest}")
                            else:
                                st.write("No interests matched")
                            
                            st.write("**Reasoning:**")
                            st.write(reasoning)
                        
                        elif selected_use_case == "content_summarizer":
                            # First check if it's an ArticleSummary object
                            if isinstance(judge_decision, ArticleSummary):
                                summary = judge_decision.summary
                                key_points = judge_decision.key_points
                                tech_details = judge_decision.technical_details
                            else:
                                # Fallback to dictionary access for backward compatibility
                                summary = judge_decision.get('summary', 'No summary provided')
                                key_points = judge_decision.get('key_points', [])
                                tech_details = judge_decision.get('technical_details', [])
                            
                            st.write("**Generated Summary:**")
                            st.markdown(summary)
                            
                            st.write("**Key Points:**")
                            if key_points:
                                for point in key_points:
                                    st.write(f"- {point}")
                            
                            st.write("**Technical Details:**")
                            if tech_details:
                                for detail in tech_details:
                                    st.write(f"- {detail}")
                    
                    # Human evaluation inputs
                    human_decision_options = ["PASS", "FAIL"] if selected_use_case == "content_filter" else ["GOOD", "NEEDS IMPROVEMENT"]
                    
                    eval_id = eval_entry.get('id')
                    col_rating, col_notes = st.columns([1, 3])
                    
                    with col_rating:
                        human_decision = st.radio(
                            f"Your rating for item {i+1}:",
                            human_decision_options,
                            key=f"decision_{eval_id}",
                            horizontal=True
                        )
                    
                    with col_notes:
                        # Get original context from config if available
                        original_context = CRITIQUE_CONFIG.get("evaluation_context", "Evaluate if content matches user interests")
                        
                        # Add button to generate critique
                        if st.button("Generate Critique", key=f"gen_critique_{eval_id}"):
                            with st.spinner("Generating critique..."):
                                auto_critique = generate_model_critique(eval_entry, original_context)
                                st.session_state[f"critique_{eval_id}"] = auto_critique
                        
                        # Initialize the critique text area with model-generated critique if available
                        critique_text = st.session_state.get(f"critique_{eval_id}", "")
                        human_notes = st.text_area(
                            "Critique (you can edit this)",
                            value=critique_text,
                            key=f"notes_{eval_id}",
                            height=200
                        )
                    
                    # Submit button in its own row for better layout
                    if st.button("Submit Evaluation", key=f"submit_{eval_id}"):
                        # Convert decision to boolean for storage
                        decision_bool = (human_decision == "PASS" or human_decision == "GOOD")
                        
                        if judge_app.save_human_feedback(eval_id, decision_bool, human_notes):
                            st.success("Evaluation saved!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Failed to save evaluation.")
                        
                        # Add exception option for content filtering
                        if selected_use_case == "content_filter":
                            add_exception = st.checkbox("Add this article as an exception", key=f"exception_{eval_id}")
                            
                            # Exception reason
                            if add_exception:
                                if human_decision == "PASS":
                                    exception_reason = st.text_area(
                                        "Why should this specific article pass despite not matching your interests?",
                                        placeholder="This helps improve future filtering...",
                                        key=f"exception_reason_pass_{eval_id}"
                                    )
                                else:  # FAIL
                                    exception_reason = st.text_area(
                                        "Why should this specific article be excluded despite matching your interests?",
                                        placeholder="This helps improve future filtering...",
                                        key=f"exception_reason_fail_{eval_id}"
                                    )
                                
                                if st.button("Save Exception", key=f"save_exception_{eval_id}"):
                                    exception_data = {
                                        'article_id': eval_entry.get('id'),
                                        'file_path': eval_entry.get('file_path', ''),
                                        'title': eval_entry.get('title', 'Unknown'),
                                        'exception_type': 'manual_pass' if human_decision == "PASS" else 'manual_fail',
                                        'reason': exception_reason if exception_reason else "No reason provided"
                                    }
                                    
                                    if save_article_exception(exception_data):
                                        st.success("Article exception saved!")
                                        time.sleep(1)
                                    else:
                                        st.error("Failed to save exception.")
    
    # Tab 2: Statistics (formerly Tab 3)
    with tabs[1]:
        st.subheader("Evaluation Statistics")
        
        # Calculate statistics
        stats = judge_app.get_agreement_stats(use_case=selected_use_case, days=days_back)
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", stats['total'])
        
        with col2:
            st.metric("Validated Articles", stats['validated'])
        
        with col3:
            st.metric("Agreement Rate", f"{stats['agreement_rate']:.1%}")
        
        with col4:
            st.metric("False Positives", stats['false_positives'])
        
        # Agreement Rate Over Time
        st.subheader("Agreement Rate Over Time")
        # Get all validated evaluations
        validated_evals = [e for e in judge_app.get_evaluations(use_case=selected_use_case, days=days_back) 
                          if e.get('human_validated', False)]
        
        if validated_evals:
            # Create dataframe with dates and agreement info
            agreement_data = []
            for eval_entry in validated_evals:
                validation_date = eval_entry.get('human_validation_timestamp', '')
                if validation_date:
                    try:
                        date = datetime.fromisoformat(validation_date).date()
                        model_decision = eval_entry.get('judge_decision').pass_filter
                        human_decision = eval_entry.get('human_decision', False)
                        agreement = int(model_decision == human_decision)
                        agreement_data.append({
                            'date': date,
                            'agreement': agreement
                        })
                    except ValueError:
                        pass
            
            if agreement_data:
                # Convert to dataframe
                df = pd.DataFrame(agreement_data)
                # Group by date and calculate agreement rate
                daily_agreement = df.groupby('date').agg(
                    agreement_rate=('agreement', 'mean'),
                    count=('agreement', 'count')
                ).reset_index()
                
                # Create time series plot
                fig = px.line(
                    daily_agreement, 
                    x='date', 
                    y='agreement_rate',
                    title='Daily Agreement Rate',
                    labels={'date': 'Date', 'agreement_rate': 'Agreement Rate'},
                    markers=True
                )
                
                # Add hover information showing count of evaluations
                fig.update_traces(
                    hovertemplate='Date: %{x}<br>Agreement Rate: %{y:.1%}<br>Evaluations: %{customdata}<extra></extra>',
                    customdata=daily_agreement['count']
                )
                
                # Format y-axis as percentage
                fig.update_layout(yaxis=dict(tickformat='.0%'))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add 7-day rolling average if enough data
                if len(daily_agreement) > 7:
                    daily_agreement['rolling_avg'] = daily_agreement['agreement_rate'].rolling(7).mean()
                    fig2 = px.line(
                        daily_agreement.dropna(), 
                        x='date', 
                        y='rolling_avg',
                        title='7-Day Rolling Average Agreement Rate',
                        labels={'date': 'Date', 'rolling_avg': 'Agreement Rate (7-day avg)'}
                    )
                    fig2.update_layout(yaxis=dict(tickformat='.0%'))
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Not enough data with timestamps to generate agreement over time chart.")
        else:
            st.info("No validated evaluations available for agreement over time analysis.")
        
        # Topic distribution
        st.subheader("Topic Distribution")
        topics = extract_topics(evaluations)
        if topics:
            top_topics = dict(topics.most_common(15))
            fig = px.bar(
                x=list(top_topics.keys()),
                y=list(top_topics.values()),
                labels={'x': 'Topic', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available.")
        
        # Interest matches
        st.subheader("Matched Interests")
        interests = extract_interest_matches(evaluations)
        if interests:
            top_interests = dict(interests.most_common(15))
            fig = px.pie(
                values=list(top_interests.values()),
                names=list(top_interests.keys()),
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No interest match data available.")


if __name__ == "__main__":
    main()