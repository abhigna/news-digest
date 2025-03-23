import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal
import random

logger = logging.getLogger(__name__)

class EvaluationSystem:
    """Generic evaluation system that can handle multiple use cases."""
    
    def __init__(self, config: Dict):
        """
        Initialize the evaluation system.
        
        Args:
            config: Configuration for the evaluation system
        """
        self.config = config
        self.eval_dir = config.get('evaluation_logs_directory', 'evaluation_logs')
        
        # Ensure directories exist
        os.makedirs(self.eval_dir, exist_ok=True)
    
    def log_evaluation(self, 
                       use_case: str, 
                       item_id: str, 
                       item_metadata: Dict, 
                       eval_result: Dict,
                       additional_data: Optional[Dict] = None,
                       is_cached: bool = False) -> str:
        """
        Log an evaluation for any use case.
        
        Args:
            use_case: The use case (e.g., 'content_filter', 'content_summary')
            item_id: Unique identifier for the item being evaluated
            item_metadata: Metadata about the item (title, url, etc.)
            eval_result: The evaluation result
            additional_data: Any additional data to include
            is_cached: Whether this evaluation result is from cache
            
        Returns:
            str: The evaluation ID
        """
        # Generate a unique evaluation ID
        eval_id = str(uuid.uuid4())
        
        # Skip logging if the result is from cache
        if is_cached:
            logger.debug(f"Skipping log for cached {use_case} evaluation for item: {item_metadata.get('title', item_id)}")
            return eval_id
        
        # Create evaluation entry
        eval_entry = {
            'id': eval_id,
            'use_case': use_case,
            'item_id': item_id,
            'item_metadata': item_metadata,
            'evaluation_result': eval_result,
            'human_validated': False,
            'human_decision': None,
            'human_notes': None,
            'evaluation_timestamp': datetime.now().isoformat(),
        }
        
        # Add additional data if provided
        if additional_data:
            eval_entry['additional_data'] = additional_data
        
        # Save to evaluation log file
        self._save_evaluation(eval_entry)
        
        logger.info(f"Logged {use_case} evaluation for item: {item_metadata.get('title', item_id)}")
        
        return eval_id
    
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
            eval_files = [f for f in os.listdir(self.eval_dir) if f.endswith('.json')]
            
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
        
        # Get all evaluation files
        eval_files = [f for f in os.listdir(self.eval_dir) if f.endswith('.json')]
        
        for eval_file in eval_files:
            file_path = os.path.join(self.eval_dir, eval_file)
            
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
                    timestamp_str = eval_entry.get('evaluation_timestamp', '')
                    if timestamp_str:
                        try:
                            timestamp = datetime.fromisoformat(timestamp_str).timestamp()
                            if timestamp < cutoff_date:
                                continue
                        except ValueError:
                            pass
                    
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
            # This assumes there's a pass_filter field in the evaluation result
            model_decision = eval_entry.get('evaluation_result', {}).get('pass_filter', False)
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
    
    def _save_evaluation(self, eval_entry: Dict) -> bool:
        """
        Save an evaluation entry to the appropriate log file.
        
        Args:
            eval_entry: The evaluation entry to save
            
        Returns:
            bool: Success status
        """
        try:
            # Create a new log file for today if it doesn't exist
            log_file = os.path.join(
                self.eval_dir,
                f"eval_log_{datetime.now().strftime('%Y%m%d')}.json"
            )
            
            # Load existing evaluations or create new list
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    evaluations = json.load(f)
            else:
                evaluations = []
            
            # Add new evaluation
            evaluations.append(eval_entry)
            
            # Save back to file
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(evaluations, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluation: {e}")
            return False
    
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
            model_decision = eval_entry.get('evaluation_result', {}).get('pass_filter', False)
            human_decision = eval_entry.get('human_decision', False)
            
            # If there's a disagreement
            if model_decision != human_decision:
                disagreements.append(eval_entry)
        
        # Randomly select examples
        if len(disagreements) <= count:
            return disagreements
        else:
            return random.sample(disagreements, count)