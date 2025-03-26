import os
import json
import logging
from datetime import datetime
from models import (
    HumanFeedback, 
    ModelResponse, 
    ContentFilterModelResponse, 
    ContentSummaryModelResponse
)

logger = logging.getLogger(__name__)

def convert_json_logs(logs_dir: str, backup: bool = True):
    """
    Convert existing JSON log files to use the new model formats.
    
    Args:
        logs_dir: Directory containing log files
        backup: Whether to create backups of original files
    """
    logger.info(f"Converting JSON logs in {logs_dir}")
    
    # Find all JSON files
    json_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        file_path = os.path.join(logs_dir, json_file)
        
        # Skip human_feedback.json as we'll handle it separately
        if json_file == "human_feedback.json":
            convert_human_feedback(file_path, backup)
            continue
            
        try:
            # Determine type of log file
            log_type = None
            if "content_filter" in json_file:
                log_type = "filter"
            elif "content_summarizer" in json_file:
                log_type = "summary"
            else:
                log_type = "generic"
                
            # Create backup if needed
            if backup:
                backup_path = f"{file_path}.bak"
                logger.info(f"Creating backup at {backup_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
            # Load the log file
            with open(file_path, 'r', encoding='utf-8') as f:
                log_entries = json.load(f)
                
            # Process each entry
            updated_entries = []
            for entry in log_entries:
                updated_entry = convert_log_entry(entry, log_type)
                updated_entries.append(updated_entry)
                
            # Save updated entries
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(updated_entries, f, indent=2)
                
            logger.info(f"Converted {len(updated_entries)} entries in {json_file}")
                
        except Exception as e:
            logger.error(f"Error converting {json_file}: {e}")

def convert_human_feedback(file_path: str, backup: bool = True):
    """Convert human feedback file to use the new HumanFeedback model"""
    if not os.path.exists(file_path):
        logger.info(f"No human feedback file found at {file_path}")
        return
        
    try:
        # Create backup if needed
        if backup:
            backup_path = f"{file_path}.bak"
            logger.info(f"Creating backup at {backup_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        # Load the feedback file
        with open(file_path, 'r', encoding='utf-8') as f:
            feedback_entries = json.load(f)
            
        # Process each entry
        updated_entries = []
        for entry in feedback_entries:
            updated_entry = {
                "eval_id": entry.get("eval_id", ""),
                "human_decision": entry.get("human_decision", False),
                "human_notes": entry.get("human_notes", ""),
                "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                "title": entry.get("title", "Unknown"),
                # Add model_decision field with a default
                "model_decision": entry.get("model_decision", False)
            }
            updated_entries.append(updated_entry)
            
        # Save updated entries
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(updated_entries, f, indent=2)
            
        logger.info(f"Converted {len(updated_entries)} entries in human feedback file")
            
    except Exception as e:
        logger.error(f"Error converting human feedback file: {e}")

def convert_log_entry(entry: dict, log_type: str) -> dict:
    """Convert a single log entry to use the new model format"""
    # Keep original structure but update response field
    updated_entry = entry.copy()
    
    # Get the response data
    response = entry.get("response", {})
    
    if log_type == "filter" and isinstance(response, dict):
        # Convert to ContentFilterModelResponse format
        model_response = ContentFilterModelResponse(
            id=entry.get("item_id", entry.get("id", "")),
            timestamp=entry.get("timestamp", datetime.now().isoformat()),
            item_metadata=entry.get("item_metadata", {}),
            is_cached=entry.get("is_cached", False),
            pass_filter=response.get("pass_filter", False),
            main_topics=response.get("main_topics", []),
            reasoning=response.get("reasoning", ""),
            specific_interests_matched=response.get("specific_interests_matched", [])
        )
        updated_entry["response"] = model_response.model_dump()
        
    elif log_type == "summary" and isinstance(response, dict):
        # Convert to ContentSummaryModelResponse format
        model_response = ContentSummaryModelResponse(
            id=entry.get("item_id", entry.get("id", "")),
            timestamp=entry.get("timestamp", datetime.now().isoformat()),
            item_metadata=entry.get("item_metadata", {}),
            is_cached=entry.get("is_cached", False),
            summary=response.get("summary", ""),
            key_points=response.get("key_points", [])
        )
        updated_entry["response"] = model_response.model_dump()
        
    else:
        # Generic response
        model_response = ModelResponse(
            id=entry.get("item_id", entry.get("id", "")),
            timestamp=entry.get("timestamp", datetime.now().isoformat()),
            item_metadata=entry.get("item_metadata", {}),
            is_cached=entry.get("is_cached", False)
        )
        # Only update if we can't determine the type
        if not isinstance(response, dict) or not ("pass_filter" in response or "summary" in response):
            updated_entry["response"] = model_response.model_dump()
    
    return updated_entry

def load_model_response(response_data: dict) -> ModelResponse:
    """
    Load model response from dictionary data, determining the appropriate type.
    
    Args:
        response_data: Dictionary containing response data
        
    Returns:
        ModelResponse: The appropriate model response object
    """
    if "pass_filter" in response_data:
        return ContentFilterModelResponse(**response_data)
    elif "summary" in response_data:
        return ContentSummaryModelResponse(**response_data)
    else:
        return ModelResponse(**response_data)

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    logs_dir = "llm_trace_logs" if len(sys.argv) < 2 else sys.argv[1]
    backup = True if len(sys.argv) < 3 else sys.argv[2].lower() == "true"
    
    convert_json_logs(logs_dir, backup) 