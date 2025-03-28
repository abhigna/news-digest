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
from models import HumanFeedback, ModelResponse, ContentFilterModelResponse, ContentSummaryModelResponse
from data_migration import load_model_response, convert_json_logs


# Set up logging
logger = logging.getLogger(__name__)

# Constants
ARTICLES_DIR = COLLECTION_CONFIG.get("articles_directory", "articles")
FEEDBACK_FILE = JUDGE_CONFIG.get("human_feedback_file")
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
        """Initialize the judge application.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_trace_logs_dir = config.get('llm_trace_logs')
        os.makedirs(self.llm_trace_logs_dir, exist_ok=True)
        
        # Initialize the JudgeSystem
        from judge_system import JudgeSystem
        self.judge_system = JudgeSystem(config)
    
    def get_evaluations(self, use_case=None, days=30, validated_only=False):
        """Get evaluations filtered by criteria"""
        evaluations = self.judge_system.get_evaluations(
            use_case=use_case,
            days=days,
            human_validated_only=validated_only
        )
        
        # Normalize evaluations to ensure consistent format
        normalized_evals = [self.normalize_evaluation(eval_entry) for eval_entry in evaluations]
        
        return normalized_evals
    
    def save_human_feedback(self, eval_id, human_decision, human_notes=None):
        """Save human feedback on a model evaluation"""
        return self.judge_system.save_human_feedback(
            eval_id=eval_id,
            human_decision=human_decision,
            human_notes=human_notes
        )
    
    def normalize_evaluation(self, eval_entry):
        """
        Normalize evaluation data to ensure consistent format regardless of source.
        This helps handle both old and new format evaluations.
        """
        normalized = eval_entry.copy()
        
        # Handle the response field - convert to model objects if needed
        response_data = eval_entry.get('response', {})
        if isinstance(response_data, dict):
            # Convert the response to the appropriate model object
            try:
                model_response = load_model_response(response_data)
                normalized['model_response_obj'] = model_response
                
                # For backwards compatibility, keep the original response
                normalized['response'] = response_data
            except Exception as e:
                logger.error(f"Error converting response data: {e}")
        
        # Check if this evaluation has human feedback
        feedback = self._get_human_feedback(eval_entry.get('id'))
        if feedback:
            normalized['human_feedback'] = feedback
            normalized['human_validated'] = True
            normalized['human_decision'] = feedback.human_decision
            normalized['human_notes'] = feedback.human_notes
        
        return normalized
    
    def _get_human_feedback(self, eval_id):
        """Get human feedback for a specific evaluation"""
        feedback_file = self.config.get("human_feedback_file")
        
        if not os.path.exists(feedback_file):
            return None
            
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_entries = json.load(f)
            
            for entry in feedback_entries:
                if entry.get('eval_id') == eval_id:
                    # Convert dict to HumanFeedback object
                    return HumanFeedback(**entry)
        except Exception as e:
            logger.error(f"Error getting human feedback: {e}")
            
        return None
    
    def _add_to_feedback_file(self, eval_id, human_decision, human_notes):
        """Add or update entry in the human feedback file"""
        feedback_file = self.config.get("human_feedback_file")
        
        # Create file if it doesn't exist
        if not os.path.exists(feedback_file):
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
                
        # Load existing feedback
        with open(feedback_file, 'r', encoding='utf-8') as f:
            feedback_entries = json.load(f)
        
        # Find evaluation to get model decision
        model_decision = False
        evaluation = self._find_evaluation(eval_id)
        if evaluation:
            response = evaluation.get('response', {})
            model_decision = response.get('pass_filter', False)
        
        # Check if entry already exists
        for entry in feedback_entries:
            if entry.get('eval_id') == eval_id:
                # Update existing entry
                entry['human_decision'] = human_decision
                entry['human_notes'] = human_notes
                entry['timestamp'] = datetime.now().isoformat()
                entry['model_decision'] = model_decision
                
                # Save updates
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump(feedback_entries, f, indent=2)
                return True
        
        # Create new entry
        new_entry = HumanFeedback(
            eval_id=eval_id,
            human_decision=human_decision,
            human_notes=human_notes or "",
            model_decision=model_decision,
            timestamp=datetime.now().isoformat(),
            title=self._get_title_for_eval(eval_id)
        ).model_dump()
        
        feedback_entries.append(new_entry)
        
        # Save updates
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_entries, f, indent=2)
            
        return True
    
    def _find_evaluation(self, eval_id):
        """Find evaluation by ID"""
        # Get all evaluations
        all_evals = self.get_evaluations(days=365)  # Look back a full year
        
        for eval_entry in all_evals:
            if eval_entry.get('id') == eval_id:
                return eval_entry
                
        return None
    
    def _get_title_for_eval(self, eval_id):
        """Get title for an evaluation"""
        eval_entry = self._find_evaluation(eval_id)
        if eval_entry:
            return eval_entry.get('item_metadata', {}).get('title', 'Unknown')
        return 'Unknown'
    
    def get_agreement_stats(self, use_case=None, days=30):
        """Calculate agreement statistics."""
        # Get all evaluations for the period first to get the correct total
        all_evaluations = self.get_evaluations(use_case=use_case, days=days)

        stats = {
            'total': len(all_evaluations),
            'validated': 0,
            'agreement': 0,
            'agreement_rate': 0,
            'false_positives': 0,  # Model says PASS, Human says FAIL
            'false_negatives': 0   # Model says FAIL, Human says PASS
        }

        # Filter for validated evaluations
        validated_evals = [e for e in all_evaluations if e.get('human_validated', False)]
        stats['validated'] = len(validated_evals)

        if stats['validated'] == 0:
            # No validated evaluations, agreement rate is undefined (or 0)
            return stats

        comparable_count = 0 # Counter for evaluations where comparison is meaningful (e.g., content filters)

        # Calculate agreement metrics only on validated evaluations
        for eval_entry in validated_evals:
            # Get the raw model response dictionary
            response = eval_entry.get('response', {})
            # Get the human decision (should exist since it's validated)
            human_decision = eval_entry.get('human_decision', None)

            # Check if this evaluation is a content filter type and has a human decision
            # We can only calculate agreement for types that have a comparable boolean decision
            if 'pass_filter' in response and human_decision is not None:
                # Extract the model's boolean decision from the 'response' dictionary
                model_decision = response.get('pass_filter', False) # Default to False if key exists but value is missing

                comparable_count += 1 # This evaluation can be compared

                # Compare model and human decisions
                if model_decision == human_decision:
                    stats['agreement'] += 1
                elif model_decision and not human_decision:
                    # Model passed, but human failed -> False Positive
                    stats['false_positives'] += 1
                elif not model_decision and human_decision:
                    # Model failed, but human passed -> False Negative
                    stats['false_negatives'] += 1
            # else: This validated evaluation might be a summary, or lack human_decision data; skip comparison.

        # Calculate agreement rate based only on the evaluations that were comparable
        if comparable_count > 0:
            stats['agreement_rate'] = stats['agreement'] / comparable_count
        # If comparable_count is 0 (e.g., only validated summaries), agreement rate remains 0

        return stats
        
    def get_time_based_stats(self, use_case=None, days=30, interval='day'):
        """
        Get agreement statistics over time.
        
        Args:
            use_case: Optional filter by use case
            days: Number of days to include
            interval: Time interval for grouping ('day', 'week', or 'month')
            
        Returns:
            DataFrame with time-based statistics
        """
        # Get all evaluations for the period
        all_evaluations = self.get_evaluations(use_case=use_case, days=days)
        
        # Filter for validated evaluations with content filter decisions
        validated_evals = []
        for eval_entry in all_evaluations:
            if eval_entry.get('human_validated', False):
                response = eval_entry.get('response', {})
                human_decision = eval_entry.get('human_decision', None)
                
                # Only include evaluations with pass_filter and human_decision
                if 'pass_filter' in response and human_decision is not None:
                    validated_evals.append(eval_entry)
        
        if not validated_evals:
            # Return empty DataFrame if no validated evaluations
            return pd.DataFrame(columns=['date', 'agreement_rate', 'false_positives', 'false_negatives', 'total'])
        
        # Convert evaluations to DataFrame for time-based analysis
        data = []
        for eval_entry in validated_evals:
            timestamp_str = eval_entry.get('timestamp', '')
            if not timestamp_str:
                continue
                
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                response = eval_entry.get('response', {})
                model_decision = response.get('pass_filter', False)
                human_decision = eval_entry.get('human_decision', False)
                
                # Determine agreement type
                if model_decision == human_decision:
                    agreement_type = 'agreement'
                elif model_decision and not human_decision:
                    agreement_type = 'false_positive'
                else:
                    agreement_type = 'false_negative'
                
                data.append({
                    'timestamp': timestamp,
                    'agreement_type': agreement_type
                })
            except ValueError:
                # Skip entries with invalid timestamps
                continue
        
        if not data:
            # Return empty DataFrame if no valid data
            return pd.DataFrame(columns=['date', 'agreement_rate', 'false_positives', 'false_negatives', 'total'])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Group by time interval
        if interval == 'day':
            df['date'] = df['timestamp'].dt.date
        elif interval == 'week':
            df['date'] = df['timestamp'].dt.to_period('W').dt.start_time.dt.date
        else:  # month
            df['date'] = df['timestamp'].dt.to_period('M').dt.start_time.dt.date
        
        # Count by date and agreement type
        counts = pd.crosstab(df['date'], df['agreement_type'])
        
        # Ensure all columns exist
        for col in ['agreement', 'false_positive', 'false_negative']:
            if col not in counts.columns:
                counts[col] = 0
        
        # Calculate totals and rates
        counts['total'] = counts.sum(axis=1)
        counts['agreement_rate'] = counts['agreement'] / counts['total']
        counts['false_positives'] = counts['false_positive']
        counts['false_negatives'] = counts['false_negative']
        
        # Reset index to make date a column
        result = counts.reset_index()
        
        # Select and rename columns
        result = result[['date', 'agreement_rate', 'false_positives', 'false_negatives', 'total']]
        
        return result


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
    """Generate a critique of the model's decision"""
    try:
        # Get normalized evaluation
        if 'model_response_obj' not in eval_entry:
            # This is already normalized data
            eval_entry = judge_app.normalize_evaluation(eval_entry)
            
        # Extract information
        response = eval_entry.get('response', {})
        human_feedback = eval_entry.get('human_feedback')
        
        # Get model decision
        model_decision = response.get('pass_filter', False)  # For content filter
        
        # Check if required configuration is available
        if not CRITIQUE_CONFIG.get("api_key"):
            logger.warning("No API key found in CRITIQUE_CONFIG, falling back to simple critique")
            
            # Special case for summarizer
            if 'summary' in response:
                # This is a content summary
                return "Content summary critique not implemented yet"
            
            # Content filter critique - works without human feedback
            critique = f"## Model Decision Critique\n\n"
            critique += f"**Model Decision**: {'PASS' if model_decision else 'FAIL'}\n\n"
            
            # Add human decision if available
            if human_feedback:
                human_decision = human_feedback.human_decision
                if model_decision == human_decision:
                    agreement = "The model's decision matches the human evaluation."
                else:
                    agreement = "The model's decision does NOT match the human evaluation."
                    
                critique += f"**Human Decision**: {'PASS' if human_decision else 'FAIL'}\n\n"
                critique += f"**Agreement**: {agreement}\n\n"
            
            # Add reasoning
            critique += f"**Model Reasoning**:\n{response.get('reasoning', 'No reasoning provided')}\n\n"
            
            # Add topics
            topics = response.get('main_topics', [])
            critique += f"**Identified Topics**:\n"
            critique += "\n".join([f"- {topic}" for topic in topics])
            critique += "\n\n"
            
            # Add human notes if available
            if human_feedback and human_feedback.human_notes:
                critique += f"**Human Notes**:\n{human_feedback.human_notes}\n\n"
            
            return critique
        
        # Get content and title for the article
        title = eval_entry.get('item_metadata', {}).get('title', 'Unknown Article')
        file_path = eval_entry.get('item_metadata', {}).get('file_path', '')
        content, _ = get_article_content(file_path)
        
        # Check if we have API configuration
        if not all([
            CRITIQUE_CONFIG.get('api_base'),
            CRITIQUE_CONFIG.get('api_key'),
            CRITIQUE_CONFIG.get('model')
        ]):
            logger.error("Incomplete API configuration for critique generation")
            return "Error: Incomplete API configuration. Please check the CRITIQUE_CONFIG settings."
            
        try:
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
Reasoning: {response.get('reasoning', 'No reasoning provided')}
Topics Identified: {', '.join(response.get('main_topics', [])) if response.get('main_topics', []) else 'None'}
Interests Matched: {', '.join(response.get('specific_interests_matched', [])) if response.get('specific_interests_matched', []) else 'None'}

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
                summary = response.get('summary', 'No summary provided')
                key_points = response.get('key_points', [])
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
            
            # Log API request for debugging
            logger.info(f"Sending critique request to API: {CRITIQUE_CONFIG['model']}")
            
            # Get simple text output instead of structured output
            api_response = client.chat.completions.create(
                model=CRITIQUE_CONFIG["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=CRITIQUE_CONFIG.get("max_tokens", 500),
                temperature=CRITIQUE_CONFIG.get("temperature", 0.7),
                timeout=CRITIQUE_CONFIG.get('timeout_seconds', 30)
            )
            
            # Check if we got a valid response
            if not api_response or not hasattr(api_response, 'choices') or not api_response.choices:
                logger.error(f"Invalid API response: {api_response}")
                return "Error: Received invalid response from the API. Please check the logs for details."
            
            # Return the simple text critique
            return api_response.choices[0].message.content
            
        except ImportError as e:
            logger.error(f"OpenAI library not found: {e}")
            return "Error: OpenAI library not installed. Please install it with 'pip install openai'."
            
        except Exception as e:
            logger.error(f"Error in critique generation: {e}")
            return f"Error generating critique: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error generating model critique: {e}")
        return f"Error generating critique: {str(e)}"


def run_interface(judge_app):
    """Run the Streamlit interface for the judge application"""
    st.set_page_config(
        page_title="Content Judge System",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Content Judge System")
    st.sidebar.title("Navigation")
    
    # Navigation options
    page = st.sidebar.radio(
        "Select Page",
        ["Evaluation Dashboard", "Review Content", "Interest Management", "Exceptions Management"]
    )
    
    # Display the selected page
    if page == "Evaluation Dashboard":
        show_dashboard(judge_app)
    elif page == "Review Content":
        show_review_page(judge_app)
    elif page == "Interest Management":
        show_interest_management()
    elif page == "Exceptions Management":
        show_exceptions_management()


def show_dashboard(judge_app):
    """Show evaluation dashboard with statistics"""
    st.header("Evaluation Dashboard")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        use_case = st.selectbox(
            "Filter by Use Case",
            ["All", "content_filter", "content_summarizer"]
        )
    with col2:
        days = st.slider("Days to Include", 1, 90, 30)
    with col3:
        interval = st.selectbox(
            "Time Interval",
            ["day", "week", "month"]
        )
    
    # Get filtered stats
    use_case_filter = None if use_case == "All" else use_case
    stats = judge_app.get_agreement_stats(use_case=use_case_filter, days=days)
    
    # Display metrics
    metrics_cols = st.columns(5)
    metrics_cols[0].metric("Total Evaluations", stats['total'])
    metrics_cols[1].metric("Validated", stats['validated'])
    metrics_cols[2].metric("Agreement Rate", f"{stats['agreement_rate']:.2%}")
    metrics_cols[3].metric("False Positives", stats['false_positives'])
    metrics_cols[4].metric("False Negatives", stats['false_negatives'])
    
    # Get time-based stats for charts
    time_stats = judge_app.get_time_based_stats(use_case=use_case_filter, days=days, interval=interval)
    
    # Display charts if we have data
    if not time_stats.empty:
        st.subheader(f"Stats Over Time (by {interval})")
        
        # Create tabs for different charts
        tab1, tab2, tab3 = st.tabs(["Agreement Rate", "False Positives", "False Negatives"])
        
        with tab1:
            # Agreement Rate chart
            fig_agreement = px.bar(
                time_stats, 
                x='date', 
                y='agreement_rate',
                title='Agreement Rate Over Time',
                labels={'date': 'Date', 'agreement_rate': 'Agreement Rate'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_agreement.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_agreement, use_container_width=True)
        
        with tab2:
            # False Positives chart
            fig_fp = px.bar(
                time_stats, 
                x='date', 
                y='false_positives',
                title='False Positives Over Time',
                labels={'date': 'Date', 'false_positives': 'False Positives'},
                color_discrete_sequence=['#ff7f0e']
            )
            st.plotly_chart(fig_fp, use_container_width=True)
        
        with tab3:
            # False Negatives chart
            fig_fn = px.bar(
                time_stats, 
                x='date', 
                y='false_negatives',
                title='False Negatives Over Time',
                labels={'date': 'Date', 'false_negatives': 'False Negatives'},
                color_discrete_sequence=['#d62728']
            )
            st.plotly_chart(fig_fn, use_container_width=True)
    
    # Show recent evaluations
    st.subheader("Recent Evaluations")
    evaluations = judge_app.get_evaluations(use_case=use_case_filter, days=days)
    
    if not evaluations:
        st.info("No evaluations found with the current filters.")
        return
    
    # Create a DataFrame for display
    eval_data = []
    for eval_entry in evaluations:
        # Extract response data
        response = eval_entry.get('response', {})
        
        # Initialize variables with defaults
        entry_type = "Unknown"
        decision = "N/A"  # Default value for decision
        
        # Determine if it's a filter or summary
        if isinstance(response, dict):
            if 'pass_filter' in response:
                entry_type = "Filter"
                decision = "PASS" if response.get('pass_filter') else "FAIL"
            elif 'summary' in response:
                entry_type = "Summary"
                # decision stays as "N/A"
        
        # Get human validation info
        human_validated = eval_entry.get('human_validated', False)
        human_decision = "PASS" if eval_entry.get('human_decision', False) else "FAIL" if human_validated else "N/A"
        
        # Add to data
        eval_data.append({
            'ID': eval_entry.get('id', ''),
            'Timestamp': eval_entry.get('timestamp', ''),
            'Title': eval_entry.get('item_metadata', {}).get('title', 'Unknown'),
            'Type': entry_type,
            'Model Decision': decision,
            'Human Decision': human_decision,
            'Validated': "Yes" if human_validated else "No"
        })
    
    # Show as dataframe
    df = pd.DataFrame(eval_data)
    st.dataframe(df)


def show_review_page(judge_app):
    """Show content review page"""
    st.header("Review Content")
    
    # Filter controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        use_case = st.selectbox(
            "Filter by Use Case",
            ["All", "content_filter", "content_summarizer"],
            key="review_use_case"
        )
    with col2:
        days = st.slider("Days to Include", 1, 90, 30, key="review_days")
    with col3:
        show_validated = st.checkbox("Show Validated Only", False)
    with col4:
        exclude_validated = st.checkbox("Exclude Validated", False)
    
    # Get evaluations
    use_case_filter = None if use_case == "All" else use_case
    evaluations = judge_app.get_evaluations(
        use_case=use_case_filter, 
        days=days,
        validated_only=show_validated
    )
    
    # Filter out validated examples if exclude_validated is checked
    if exclude_validated:
        evaluations = [e for e in evaluations if not e.get('human_validated', False)]
    
    if not evaluations:
        st.info("No evaluations found with the current filters.")
        return
    
    # Create a selection list of titles with model decision
    titles = []
    for e in evaluations:
        title = e.get('item_metadata', {}).get('title', 'Unknown')
        eval_id = e.get('id', '')
        
        # Get model decision
        response = e.get('response', {})
        model_decision = "N/A"
        if isinstance(response, dict):
            if 'pass_filter' in response:
                model_decision = "PASS" if response.get('pass_filter') else "FAIL"
        
        # Format title with model decision
        titles.append(f"{title} - Model: {model_decision} ({eval_id})")
    
    selected_title = st.selectbox("Select Content to Review", titles)
    
    if selected_title:
        # Extract evaluation ID from selection
        eval_id = selected_title.split('(')[-1].strip(')')
        
        # Find the selected evaluation
        selected_eval = next((e for e in evaluations if e.get('id') == eval_id), None)
        
        if selected_eval:
            display_evaluation(selected_eval, judge_app)


def display_evaluation(eval_entry, judge_app):
    """Display a single evaluation with review controls"""
    st.subheader(eval_entry.get('item_metadata', {}).get('title', 'Unknown'))
    
    # Get content
    file_path = eval_entry.get('item_metadata', {}).get('file_path', '')
    content, metadata = get_article_content(file_path)
    
    # Display basic info with model decision
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
        st.write(f"**Date:** {metadata.get('pubDate', 'Unknown')}")
    with col2:
        st.write(f"**Evaluation ID:** {eval_entry.get('id', '')}")
        st.write(f"**Evaluated:** {eval_entry.get('timestamp', '')}")
    with col3:
        # Extract and display model decision prominently
        response = eval_entry.get('response', {})
        model_decision = "N/A"
        if isinstance(response, dict):
            if 'pass_filter' in response:
                model_decision = "PASS" if response.get('pass_filter') else "FAIL"
                
        st.markdown(f"**Model Decision:** **:{'green' if model_decision == 'PASS' else 'red'}[{model_decision}]**")
        
        # Display human decision if available
        human_validated = eval_entry.get('human_validated', False)
        if human_validated:
            human_decision = "PASS" if eval_entry.get('human_decision', False) else "FAIL"
            st.markdown(f"**Human Decision:** **:{'green' if human_decision == 'PASS' else 'red'}[{human_decision}]**")
    
    # Display content
    with st.expander("Article Content", expanded=False):
        st.markdown(content)
    
    # Display model evaluation
    st.subheader("Model Evaluation")
    
    # Determine type of evaluation
    if 'pass_filter' in response:
        # Content filter response
        st.write("**Reasoning:**")
        st.markdown(response.get('reasoning', 'No reasoning provided'))
        
        st.write("**Main Topics:**")
        for topic in response.get('main_topics', []):
            st.markdown(f"- {topic}")
            
        if response.get('specific_interests_matched', []):
            st.write("**Interests Matched:**")
            for interest in response.get('specific_interests_matched', []):
                st.markdown(f"- {interest}")
    
    elif 'summary' in response:
        # Content summary response
        st.write("**Summary:**")
        st.markdown(response.get('summary', 'No summary provided'))
        
        st.write("**Key Points:**")
        for point in response.get('key_points', []):
            st.markdown(f"- {point}")
    
    # Human feedback section
    st.subheader("Human Feedback")
    
    # Check if already validated
    human_validated = eval_entry.get('human_validated', False)
    current_decision = eval_entry.get('human_decision', False) if human_validated else None
    current_notes = eval_entry.get('human_notes', '')
    
    # Decision input
    new_decision = st.radio(
        "Your Decision",
        ["PASS", "FAIL"],
        index=0 if current_decision else 1,
        help="Indicate whether this content should pass or fail the filter"
    )
    
    # Add button to generate critique
    if st.button("Generate Model Critique"):
        with st.spinner("Generating critique..."):
            critique = generate_model_critique(eval_entry)
            st.session_state[f"critique_{eval_entry.get('id')}"] = critique
    
    # Notes input with critique
    critique_text = st.session_state.get(f"critique_{eval_entry.get('id')}", current_notes)
    new_notes = st.text_area(
        "Notes (explain your decision)",
        value=critique_text,
        help="Provide your reasoning for this decision, especially if you disagree with the model"
    )
    
    # Submit button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Submit Feedback"):
            # Convert PASS/FAIL to boolean
            decision_bool = (new_decision == "PASS")
            
            # Save feedback
            if judge_app.save_human_feedback(eval_entry.get('id', ''), decision_bool, new_notes):
                st.success("Feedback saved successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to save feedback. Please try again.")


def show_interest_management():
    """Show interest management page"""
    st.header("Interest Management")
    
    # Display current interests
    st.subheader("Current Interests")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Interests**")
        for interest in INTERESTS.get("primary_interests", []):
            st.markdown(f"- {interest}")
    
    with col2:
        st.write("**Excluded Interests**")
        for interest in INTERESTS.get("excluded_interests", []):
            st.markdown(f"- {interest}")
    
    # Add new interests
    st.subheader("Add New Interests")
    new_interest = st.text_input("New Interest")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add to Primary"):
            if new_interest:
                if save_config_changes(interests_to_add=[new_interest]):
                    st.success(f"Added '{new_interest}' to primary interests")
                    time.sleep(1)
                    st.rerun()
    
    with col2:
        if st.button("Add to Excluded"):
            if new_interest:
                if save_config_changes(interests_to_remove=[new_interest]):
                    st.success(f"Added '{new_interest}' to excluded interests")
                    time.sleep(1)
                    st.rerun()


def show_exceptions_management():
    """Show exceptions management page"""
    st.header("Exceptions Management")
    
    # Get all exceptions
    exceptions = get_article_exceptions()
    
    if not exceptions:
        st.info("No article exceptions found.")
    else:
        # Display exceptions
        st.subheader("Current Exceptions")
        
        for i, exception in enumerate(exceptions):
            with st.expander(f"Exception: {exception.get('article_id', 'Unknown')}"):
                st.write(f"**Article ID:** {exception.get('article_id', 'Unknown')}")
                st.write(f"**Reason:** {exception.get('reason', 'No reason provided')}")
                st.write(f"**Date Added:** {exception.get('timestamp', 'Unknown')}")
                
                if st.button("Remove Exception", key=f"remove_{i}"):
                    if remove_exception(exception.get('article_id')):
                        st.success("Exception removed successfully!")
                        time.sleep(1)
                        st.rerun()
    
    # Add new exception
    st.subheader("Add New Exception")
    article_id = st.text_input("Article ID")
    reason = st.text_area("Reason for Exception")
    
    if st.button("Add Exception"):
        if article_id and reason:
            exception_data = {
                "article_id": article_id,
                "reason": reason
            }
            
            if save_article_exception(exception_data):
                st.success("Exception added successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to add exception. Please try again.")


def main():
    """Main entry point for the application"""
    import argparse
    from data_migration import convert_json_logs
    
    parser = argparse.ArgumentParser(description='Judge system for evaluating content filtering and summarization')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--migrate', action='store_true', help='Migrate existing data to new model format')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Migrate data if requested
    if args.migrate:
        logger.info("Migrating existing data...")
        convert_json_logs(JUDGE_CONFIG.get('llm_trace_logs', 'llm_trace_logs'))
    
    # Initialize the judge system
    global judge_app
    judge_app = JudgeApp(JUDGE_CONFIG)
    
    # Start the interface
    run_interface(judge_app)


if __name__ == "__main__":
    main()