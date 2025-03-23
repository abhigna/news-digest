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
from config import EVALUATION_CONFIG, COLLECTION_CONFIG, INTERESTS

# Set up logging
logger = logging.getLogger(__name__)

# Constants
EVAL_LOGS_DIR = EVALUATION_CONFIG.get("evaluation_logs_directory", "evaluation_logs")
ARTICLES_DIR = COLLECTION_CONFIG.get("articles_directory", "articles")
FEEDBACK_FILE = "human_feedback.json"

# Ensure directories exist
os.makedirs(EVAL_LOGS_DIR, exist_ok=True)
os.makedirs(ARTICLES_DIR, exist_ok=True)

# Initialize NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class SimpleEvaluationSystem:
    """A simplified evaluation system for managing content evaluations."""
    
    def __init__(self, config):
        self.config = config
        self.eval_dir = config.get('evaluation_logs_directory', 'evaluation_logs')
        os.makedirs(self.eval_dir, exist_ok=True)
    
    def get_evaluations(self, use_case=None, days=30, validated_only=False):
        """Get evaluations with filtering options."""
        filtered_evals = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Get evaluation files
        if not os.path.exists(self.eval_dir):
            return []
            
        eval_files = [f for f in os.listdir(self.eval_dir) if f.endswith('.json') and f != "article_exceptions.json" and f != FEEDBACK_FILE]
        
        for eval_file in eval_files:
            file_path = os.path.join(self.eval_dir, eval_file)
            
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
            eval_files = [f for f in os.listdir(self.eval_dir) if f.endswith('.json') and f != "article_exceptions.json" and f != FEEDBACK_FILE]
            
            for eval_file in eval_files:
                file_path = os.path.join(self.eval_dir, eval_file)
                
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
            # Get the evaluation result - handle different formats
            evaluation = eval_entry.get('evaluation', {})
            
            # Get decisions
            model_decision = evaluation.get('pass_filter', False)
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
            
        # Ensure evaluation data is properly structured
        if 'evaluation' not in normalized and 'evaluation_result' in normalized:
            normalized['evaluation'] = normalized['evaluation_result']
            
        return normalized
    
    def _add_to_feedback_file(self, eval_id, human_decision, human_notes):
        """Add feedback to a separate feedback file for easier analysis."""
        feedback_path = os.path.join(self.eval_dir, FEEDBACK_FILE)
        
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
        
        eval_files = [f for f in os.listdir(self.eval_dir) if f.endswith('.json') and f != "article_exceptions.json" and f != FEEDBACK_FILE]
        
        for eval_file in eval_files:
            file_path_full = os.path.join(self.eval_dir, eval_file)
            
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
        exceptions_file = os.path.join(EVAL_LOGS_DIR, "article_exceptions.json")
        
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
        exceptions_file = os.path.join(EVAL_LOGS_DIR, "article_exceptions.json")
        
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
        exceptions_file = os.path.join(EVAL_LOGS_DIR, "article_exceptions.json")
        
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
        topics = eval_entry.get('evaluation', {}).get('main_topics', [])
        topics_counter.update(topics)
    
    return topics_counter


def extract_interest_matches(evaluations):
    """Extract and count interests matched from evaluations."""
    interest_counter = Counter()
    
    for eval_entry in evaluations:
        interests = eval_entry.get('evaluation', {}).get('specific_interests_matched', [])
        interest_counter.update(interests)
    
    return interest_counter


def generate_word_cloud(evaluations):
    """Generate word count data from evaluation reasoning."""
    stop_words = set(stopwords.words('english'))
    words = []
    
    for eval_entry in evaluations:
        reasoning = eval_entry.get('evaluation', {}).get('reasoning', '')
        if reasoning:
            tokens = word_tokenize(reasoning.lower())
            words.extend([word for word in tokens if word.isalpha() and word not in stop_words])
    
    return Counter(words)


def main():
    """Main function for the Streamlit app."""
    st.set_page_config(page_title="Content Evaluation System", layout="wide")
    
    st.title("Content Evaluation System")
    
    # Initialize evaluation system
    eval_system = SimpleEvaluationSystem(EVALUATION_CONFIG)
    
    # Sidebar with use case selection
    st.sidebar.header("Options")
    
    # For now, we only have content filter
    use_cases = ["content_filter"]
    selected_use_case = st.sidebar.selectbox("Evaluation Use Case", use_cases)
    
    # Other sidebar options
    days_back = st.sidebar.slider("Days to include", 1, 90, EVALUATION_CONFIG.get("default_days_to_display", 30))
    show_validated = st.sidebar.checkbox("Show already validated items", False)
    
    # Filter options for content filter
    filter_mode = st.sidebar.radio("Filter by", ["All", "Passed filter", "Failed filter"])
    
    # Interest management in sidebar
    with st.sidebar.expander("Manage Interests", expanded=False):
        st.write("Current Primary Interests:")
        for interest in INTERESTS.get("primary_interests", []):
            st.write(f"- {interest}")
        
        st.write("Current Excluded Interests:")
        for interest in INTERESTS.get("excluded_interests", []):
            st.write(f"- {interest}")
        
        st.write("---")
        new_interest = st.text_area("Add new interest:", placeholder="Describe a new interest to add...")
        if st.button("Add to Primary Interests") and new_interest:
            if save_config_changes(interests_to_add=[new_interest]):
                st.success(f"Added '{new_interest}' to primary interests!")
                time.sleep(1)
                st.rerun()
        
        exclude_interest = st.text_area("Add to excluded interests:", placeholder="Describe an interest to exclude...")
        if st.button("Add to Excluded Interests") and exclude_interest:
            if save_config_changes(interests_to_remove=[exclude_interest]):
                st.success(f"Added '{exclude_interest}' to excluded interests!")
                time.sleep(1)
                st.rerun()
    
    # Load evaluations
    evaluations = eval_system.get_evaluations(use_case=selected_use_case, days=days_back)
    
    # Apply filters
    if filter_mode == "Passed filter":
        evaluations = [e for e in evaluations if e.get('evaluation', {}).get('pass_filter', False)]
    elif filter_mode == "Failed filter":
        evaluations = [e for e in evaluations if not e.get('evaluation', {}).get('pass_filter', False)]
    
    if not show_validated:
        evaluations = [e for e in evaluations if not e.get('human_validated', False)]
    
    # Create tabs
    tabs = st.tabs(["Evaluate Articles", "Statistics", "Exceptions"])
    
    # Tab 1: Evaluate Articles
    with tabs[0]:
        if not evaluations:
            st.info("No articles found matching the current filters.")
        else:
            st.write(f"Found {len(evaluations)} articles to evaluate.")
            
            # Create a title for each evaluation
            article_titles = []
            for e in evaluations:
                title = e.get('title', '')
                if not title:
                    title = e.get('item_metadata', {}).get('title', '')
                if not title:
                    title = f"Unknown Title ({e.get('id', 'unknown id')})"
                article_titles.append(title)
            
            # Select article to evaluate
            selected_article = st.selectbox("Select article to evaluate:", article_titles)
            selected_index = article_titles.index(selected_article)
            eval_entry = evaluations[selected_index]
            
            # Display article information
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Article Content")
                file_path = eval_entry.get('file_path', '')
                content, metadata = get_article_content(file_path)
                
                if content:
                    # Display metadata
                    st.write(f"**Title:** {metadata.get('title', 'Unknown')}")
                    st.write(f"**Source:** {metadata.get('source', 'Unknown')}")
                    st.write(f"**Date:** {metadata.get('pubDate', 'Unknown')}")
                    
                    # Display content
                    with st.expander("View article content", expanded=False):
                        st.markdown(content)
                else:
                    st.error(f"Article file not found: {file_path}")
            
            with col2:
                st.subheader("Model Evaluation")
                evaluation = eval_entry.get('evaluation', {})
                
                model_decision = "✅ PASS" if evaluation.get('pass_filter', False) else "❌ FAIL"
                st.write(f"**Model Decision:** {model_decision}")
                
                st.write("**Topics:**")
                topics = evaluation.get('main_topics', [])
                if topics:
                    for topic in topics:
                        st.write(f"- {topic}")
                else:
                    st.write("No topics available")
                
                st.write("**Interests Matched:**")
                interests = evaluation.get('specific_interests_matched', [])
                if interests:
                    for interest in interests:
                        st.write(f"- {interest}")
                else:
                    st.write("No interests matched")
                
                st.write("**Reasoning:**")
                reasoning = evaluation.get('reasoning', 'No reasoning provided')
                st.write(reasoning)
            
            # Human evaluation form
            st.subheader("Your Evaluation")
            
            with st.form("human_eval_form"):
                human_decision = st.radio(
                    "Does this article pass your filter?",
                    ["PASS", "FAIL"],
                    horizontal=True
                )
                
                if human_decision == "PASS" and not evaluation.get('pass_filter', False):
                    st.info("You're passing an article that the model rejected. Consider adding interests to your profile.")
                
                elif human_decision == "FAIL" and evaluation.get('pass_filter', False):
                    st.info("You're rejecting an article that the model passed. Consider adding exclusions to your profile.")
                
                human_notes = st.text_area(
                    "Notes (Why did you make this decision?)",
                    height=100
                )
                
                # Add as exception option
                add_exception = st.checkbox("Add this article as an exception")
                
                # Exception reason
                exception_reason = ""
                if add_exception:
                    if human_decision == "PASS":
                        exception_reason = st.text_area(
                            "Why should this specific article pass despite not matching your interests?",
                            placeholder="This helps improve future filtering..."
                        )
                    else:  # FAIL
                        exception_reason = st.text_area(
                            "Why should this specific article be excluded despite matching your interests?",
                            placeholder="This helps improve future filtering..."
                        )
                
                submit_button = st.form_submit_button("Submit Evaluation")
                
                if submit_button:
                    # Save human feedback
                    if eval_system.save_human_feedback(eval_entry.get('id'), human_decision == "PASS", human_notes):
                        st.success("Evaluation saved successfully!")
                        
                        # Save exception if requested
                        if add_exception:
                            exception_data = {
                                'article_id': eval_entry.get('id'),
                                'file_path': eval_entry.get('file_path', ''),
                                'title': eval_entry.get('title', 'Unknown'),
                                'exception_type': 'manual_pass' if human_decision == "PASS" else 'manual_fail',
                                'reason': exception_reason if exception_reason else "No reason provided"
                            }
                            
                            if save_article_exception(exception_data):
                                st.success("Article exception saved!")
                        
                        # Refresh the page after 2 seconds
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Failed to save evaluation.")
    
    # Tab 2: Statistics
    with tabs[1]:
        st.subheader("Evaluation Statistics")
        
        # Calculate statistics
        stats = eval_system.get_agreement_stats(use_case=selected_use_case, days=days_back)
        
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
        validated_evals = [e for e in eval_system.get_evaluations(use_case=selected_use_case, days=days_back) 
                          if e.get('human_validated', False)]
        
        if validated_evals:
            # Create dataframe with dates and agreement info
            agreement_data = []
            for eval_entry in validated_evals:
                validation_date = eval_entry.get('human_validation_timestamp', '')
                if validation_date:
                    try:
                        date = datetime.fromisoformat(validation_date).date()
                        model_decision = eval_entry.get('evaluation', {}).get('pass_filter', False)
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
    
    # Tab 3: Exceptions
    with tabs[2]:
        st.subheader("Article Exceptions")
        
        exceptions = get_article_exceptions()
        if not exceptions:
            st.info("No article exceptions have been created yet.")
        else:
            st.write(f"Found {len(exceptions)} article exceptions.")
            
            # Group exceptions by type
            manual_passes = [e for e in exceptions if e.get('exception_type') == 'manual_pass']
            manual_fails = [e for e in exceptions if e.get('exception_type') == 'manual_fail']
            
            if manual_passes:
                st.subheader("Manual Passes")
                st.write("These articles were manually passed despite not matching your interests:")
                
                for exception in manual_passes:
                    with st.expander(f"{exception.get('title', 'Unknown Article')}"):
                        st.write(f"**Reason:** {exception.get('reason', 'No reason provided')}")
                        st.write(f"**Date added:** {exception.get('timestamp', 'Unknown')[:10]}")
                        
                        if st.button(f"Remove exception", key=f"remove_{exception.get('article_id')}"):
                            if remove_exception(exception.get('article_id')):
                                st.success("Exception removed!")
                                time.sleep(1)
                                st.rerun()
            
            if manual_fails:
                st.subheader("Manual Fails")
                st.write("These articles were manually failed despite matching your interests:")
                
                for exception in manual_fails:
                    with st.expander(f"{exception.get('title', 'Unknown Article')}"):
                        st.write(f"**Reason:** {exception.get('reason', 'No reason provided')}")
                        st.write(f"**Date added:** {exception.get('timestamp', 'Unknown')[:10]}")
                        
                        if st.button(f"Remove exception", key=f"remove_{exception.get('article_id')}"):
                            if remove_exception(exception.get('article_id')):
                                st.success("Exception removed!")
                                time.sleep(1)
                                st.rerun()


if __name__ == "__main__":
    main()