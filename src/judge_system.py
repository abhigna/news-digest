import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
import random
import glob
from models import HumanFeedback, ModelResponse, ContentFilterModelResponse, ContentSummaryModelResponse

logger = logging.getLogger(__name__)

class JudgeSystem:
    """Generic evaluation system that can handle multiple use cases."""
    
    def __init__(self, config: Dict):
        """
        Initialize the evaluation system.
        
        Args:
            config: Configuration for the evaluation system
        """
        self.config = config
        self.llm_trace_logs_dir = config.get('llm_trace_logs', 'llm_trace_logs')
        
        # Ensure directories exist
        os.makedirs(self.llm_trace_logs_dir, exist_ok=True)
    
    def save_human_feedback(self, 
                           eval_id: str, 
                           human_decision: bool, 
                           human_notes: Optional[str] = None) -> bool:
        """
        Save human feedback for an evaluation.
        
        Args:
            eval_id: ID of the evaluation
            human_decision: Human decision (True for pass, False for fail)
            human_notes: Optional notes from human reviewer
            
        Returns:
            bool: Success status
        """
        try:
            updated = False
            
            # Find the evaluation in log files
            feedback_file = self.config.get("human_feedback_file")
            
            # Create file if it doesn't exist
            if not os.path.exists(feedback_file):
                with open(feedback_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2)
            
            # Load existing feedback
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_entries = json.load(f)
            
            # Look for existing entry
            for entry in feedback_entries:
                if entry.get('eval_id') == eval_id:
                    # Update existing entry
                    entry['human_decision'] = human_decision
                    entry['human_notes'] = human_notes
                    entry['timestamp'] = datetime.now().isoformat()
                    updated = True
                    break
            
            # Add new entry if not found
            if not updated:
                new_entry = HumanFeedback(
                    eval_id=eval_id,
                    human_decision=human_decision,
                    human_notes=human_notes or "",
                    model_decision=self._get_model_decision(eval_id)
                ).model_dump()
                feedback_entries.append(new_entry)
            
            # Save back to file
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_entries, f, indent=2)
            
            logger.info(f"Saved human feedback for evaluation {eval_id}")
            return True
                
        except Exception as e:
            logger.error(f"Error saving human feedback: {e}")
            return False
    
    def _get_model_decision(self, eval_id: str) -> bool:
        """Get the model's decision for a given evaluation ID"""
        # Find the evaluation in log files
        eval_files = [f for f in os.listdir(self.llm_trace_logs_dir) if f.endswith('.json')]
        
        for eval_file in eval_files:

            file_path = os.path.join(self.llm_trace_logs_dir, eval_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
                
                # Look for the evaluation entry
                for eval_entry in evaluations:
                    if eval_entry.get('id') == eval_id:
                        response = eval_entry.get('response', {})
                        return response.get('pass_filter', False)
            except Exception:
                pass
        
        return False
    
    def get_evaluations(self, 
                        use_case: Optional[str] = None, 
                        days: int = 30, 
                        human_validated_only: bool = False) -> List[Dict]:
        """
        Get evaluations filtered by criteria.
        
        Args:
            use_case: Optional filter by use case
            days: Number of days to include
            human_validated_only: Whether to return only human-validated evaluations
            
        Returns:
            list: Filtered evaluations
        """
        filtered_evals = []
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        # Get all evaluation files, filter by use_case if provided
        if use_case:
            # Use glob to find matching use case log files
            pattern = os.path.join(self.llm_trace_logs_dir, f"{use_case}_*.json")
            eval_files = [os.path.basename(f) for f in glob.glob(pattern)]
        else:
            eval_files = [f for f in os.listdir(self.llm_trace_logs_dir) if f.endswith('.json')]
        
        # Keep track of seen item_ids to prevent duplicates
        seen_item_ids = set()
        
        for eval_file in eval_files:
            file_path = os.path.join(self.llm_trace_logs_dir, eval_file)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
                
                for eval_entry in evaluations:
                    # Apply filters
                    if use_case and eval_entry.get('use_case') != use_case:
                        continue
                    
                    if human_validated_only and not eval_entry.get('human_validated', False):
                        continue
                    
                    # Apply date filter
                    timestamp_str = eval_entry.get('timestamp', '')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                            if timestamp < cutoff_date:
                                continue
                        except ValueError:
                            pass
                    
                    # Check for duplicates based on item_id
                    item_id = eval_entry.get('item_id')
                    if item_id and item_id in seen_item_ids:
                        # Skip this duplicate entry
                        continue
                    
                    # Add item_id to seen set
                    if item_id:
                        seen_item_ids.add(item_id)
                    
                    filtered_evals.append(eval_entry)
                    
            except Exception as e:
                logger.error(f"Error reading evaluation file {eval_file}: {e}")
        
        return filtered_evals
    
    def get_agreement_stats(self, use_case: Optional[str] = None, days: int = 30) -> Dict:
        """
        Calculate agreement statistics between model and human evaluations.
        
        Args:
            use_case: Optional filter by use case
            days: Number of days to include
            
        Returns:
            dict: Agreement statistics
        """
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
            # Get the "pass" decision from both model and human
            # Update to use the new response field structure from LlmGateway
            model_decision = eval_entry.get('response', {}).get('pass_filter', False)
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
    
    def get_disagreement_examples(self, 
                                 use_case: str,
                                 count: int = 2,
                                 days: int = 90) -> List[Dict]:
        """
        Get examples where human and model evaluations disagreed.
        
        Args:
            use_case: Filter by use case (e.g., 'content_filter')
            count: Number of examples to return
            days: Look back this many days for examples
            
        Returns:
            list: Examples with human-model disagreement
        """
        # Get human-validated evaluations for the use case
        validated_evals = self.get_evaluations(
            use_case=use_case,
            days=days,
            human_validated_only=True
        )
        
        # Filter for disagreements
        disagreements = []
        for eval_entry in validated_evals:
            # Update to use the new response field structure from LlmGateway
            model_decision = eval_entry.get('response', {}).get('pass_filter', False)
            human_decision = eval_entry.get('human_decision', False)
            
            # If there's a disagreement
            if model_decision != human_decision:
                disagreements.append(eval_entry)
        
        # Randomly select examples
        if len(disagreements) <= count:
            return disagreements
        else:
            return random.sample(disagreements, count)
    
    def get_agreement_examples(self, 
                              use_case: str,
                              count: int = 2,
                              days: int = 90) -> List[Dict]:
        """
        Get examples where human and model evaluations agreed.
        
        Args:
            use_case: Filter by use case (e.g., 'content_filter')
            count: Number of examples to return
            days: Look back this many days for examples
            
        Returns:
            list: Examples with human-model agreement
        """
        # Get human-validated evaluations for the use case
        validated_evals = self.get_evaluations(
            use_case=use_case,
            days=days,
            human_validated_only=True
        )
        
        # Filter for agreements
        agreements = []
        for eval_entry in validated_evals:
            model_decision = eval_entry.get('response', {}).get('pass_filter', False)
            human_decision = eval_entry.get('human_decision', False)
            
            # If there's an agreement
            if model_decision == human_decision:
                agreements.append(eval_entry)
        
        # Randomly select examples
        if len(agreements) <= count:
            return agreements
        else:
            return random.sample(agreements, count)
            
    def get_feedback_examples(self, use_case: str, count: int = 8) -> str:
        """
        Get formatted feedback examples for prompts.
        
        Args:
            use_case: The use case to get examples for
            count: Number of examples to include
            
        Returns:
            str: Formatted feedback examples text
        """
        # Calculate counts for disagreement and agreement examples (80/20 split)
        disagreement_count = int(count * 0.8)
        agreement_count = count - disagreement_count
        
        # Ensure at least one of each if count > 1
        if count > 1 and disagreement_count == 0:
            disagreement_count = 1
            agreement_count = count - 1
        elif count > 1 and agreement_count == 0:
            agreement_count = 1
            disagreement_count = count - 1
            
        # Get examples
        disagreement_examples = self.get_disagreement_examples(use_case=use_case, count=disagreement_count)
        agreement_examples = self.get_agreement_examples(use_case=use_case, count=agreement_count)
        
        # Combine examples
        examples = disagreement_examples + agreement_examples
        
        if not examples:
            return ""
            
        # Format examples for inclusion in prompts
        examples_text = "\nHUMAN FEEDBACK EXAMPLES (including both correct and incorrect model evaluations):\n"
        
        for i, example in enumerate(examples, 1):
            try:
                # Create HumanFeedback object
                item_meta = example.get('item_metadata', {})
                # Update to use the new response field structure
                model_result = example.get('response', {})
                model_decision = model_result.get('pass_filter', False)
                human_decision = example.get('human_decision', False)
                
                feedback = HumanFeedback(
                    eval_id=example.get('id', str(uuid.uuid4())),
                    human_decision=human_decision,
                    model_decision=model_decision,
                    human_notes=example.get('human_notes', ''),
                    title=item_meta.get('title', 'Unknown')
                )
                
                examples_text += f"Example {i}:\n"
                examples_text += feedback.format_for_prompt()
                
                # Get content snippet and limit to 250 words
                content_snippet = example.get('additional_data', {}).get('content_snippet', '')
                if content_snippet:
                    # Limit to 250 words
                    words = content_snippet.split()
                    if len(words) > 250:
                        content_snippet = ' '.join(words[:250]) + '...'
                    examples_text += f"Content snippet: {content_snippet}\n"
                
                # Add reason for model correctness/error
                if model_decision == human_decision:
                    examples_text += f"Model assessment: The model correctly assessed whether this article matched the user's interests.\n\n"
                else:
                    examples_text += f"Model assessment: The model failed to correctly assess whether this article matched the user's interests.\n\n"
            except Exception as e:
                logger.error(f"Error formatting feedback example: {e}")
        
        return examples_text