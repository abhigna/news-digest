import os
import json
import logging
import hashlib
import time
from datetime import datetime
import uuid
from typing import Dict, Any, Optional, List, Type
from pydantic import BaseModel
from openai import OpenAI
import instructor

logger = logging.getLogger(__name__)

class LlmGateway:
    """
    Gateway for LLM interactions that handles:
    - Making API calls
    - Structured output parsing
    - Caching results
    - Logging interactions to trace logs
    """
    
    def __init__(self, llm_config: Dict[str, Any], use_case: str, cache_dir: str = "llm_cache"):
        """
        Initialize the LLM gateway.
        
        Args:
            llm_config: Dictionary containing LLM configuration
            use_case: String identifier for the use case (for logging)
            cache_dir: Directory to store cache files
        """
        self.llm_config = llm_config
        self.use_case = use_case
        self.cache_dir = cache_dir
        self.trace_logs_dir = llm_config.get('llm_trace_logs', 'llm_trace_logs')
        
        # Ensure directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.trace_logs_dir, exist_ok=True)
        
        # Initialize OpenAI client
        self.base_client = OpenAI(
            base_url=llm_config.get('api_base'),
            api_key=llm_config.get('api_key')
        )
        
        # Patch with instructor for structured output
        self.client = instructor.from_openai(self.base_client, mode=instructor.Mode.JSON)
    
    def generate_response(self, 
                          prompt: str, 
                          response_model: Type[BaseModel], 
                          item_id: str = None,
                          item_metadata: Dict[str, Any] = None,
                          additional_data: Dict[str, Any] = None,
                          cache_key: str = None,
                          use_cache: bool = True) -> BaseModel:
        """
        Generate a response from the LLM with structured output.
        
        Args:
            prompt: The prompt to send to the LLM
            response_model: Pydantic model for structured output
            item_id: Unique identifier for the item being processed
            item_metadata: Metadata about the item
            additional_data: Additional data to log
            cache_key: Optional key for caching (if None, will be generated)
            use_cache: Whether to use caching
            
        Returns:
            Structured response as specified by response_model
        """
        # Generate item_id if not provided
        if not item_id:
            item_id = str(uuid.uuid4())
            
        # Generate cache key if not provided and caching is enabled
        if use_cache and not cache_key:
            cache_key = self._generate_cache_key(prompt)
        
        # Check cache first if enabled
        result = None
        is_cached = False
        if use_cache:
            result, is_cached = self._check_cache(cache_key)
            if is_cached:
                logger.info(f"Cache hit for key: {cache_key}")
        
        # If not in cache or cache disabled, call LLM
        if not result:
            try:
                result = self.client.chat.completions.create(
                    model=self.llm_config.get("model"),
                    response_model=response_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.llm_config.get("max_tokens", 1000),
                    temperature=self.llm_config.get("temperature", 0.7),
                    timeout=self.llm_config.get('timeout_seconds', 30)
                )
                
                # Cache the result if caching is enabled
                if use_cache:
                    self._save_to_cache(cache_key, result)
                    
            except Exception as e:
                logger.error(f"Error in LLM call: {e}")
                raise
        
        # Log the interaction
        self._log_interaction(
            item_id=item_id,
            prompt=prompt,
            response=result,
            item_metadata=item_metadata or {},
            additional_data=additional_data or {},
            is_cached=is_cached
        )
        
        return result
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate a cache key based on the prompt."""
        # Hash prompt and model to use as cache key
        model = self.llm_config.get("model", "unknown")
        cache_string = f"{prompt}:{model}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> tuple:
        """Check if response is in cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # If cache has model_dump (from pydantic), we need to reconstruct the object
                # For simplicity, we're just returning the raw dict here
                return cached_data, True
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None, False
    
    def _save_to_cache(self, cache_key: str, result: Any) -> None:
        """Save result to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            # Convert pydantic model to dict for storage
            if hasattr(result, "model_dump"):
                result_dict = result.model_dump()
            else:
                result_dict = result
                
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2)
            
            logger.debug(f"Saved result to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _log_interaction(self, 
                         item_id: str, 
                         prompt: str, 
                         response: Any, 
                         item_metadata: Dict[str, Any], 
                         additional_data: Dict[str, Any],
                         is_cached: bool) -> None:
        """Log the interaction to trace logs."""
        # Generate a log ID if not provided
        log_id = str(uuid.uuid4())
        
        log_entry = {
            "id": log_id,
            "use_case": self.use_case,
            "item_id": item_id,
            "item_metadata": item_metadata,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "is_cached": is_cached,
            "additional_data": additional_data
        }
        
        # Add response data with generic term
        if hasattr(response, "model_dump"):
            log_entry["response"] = response.model_dump()
        else:
            log_entry["response"] = response
        
        # Save to trace logs
        self._append_to_trace_log(log_entry)
    
    def _append_to_trace_log(self, log_entry: Dict[str, Any]) -> None:
        """Append log entry to the appropriate trace log file."""
        # Use use_case and date for log file name to keep files manageable
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.trace_logs_dir, f"{self.use_case}_{today}.json")
        
        # Read existing logs or create new file
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupt, start fresh
                logs = []
        else:
            logs = []
        
        # Append new log entry
        logs.append(log_entry)
        
        # Write back to file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2) 