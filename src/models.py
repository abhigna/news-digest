import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field


class HumanFeedback(BaseModel):
    """
    Represents human feedback on a model response
    """
    eval_id: str
    human_decision: bool = Field(description="Whether the human approved the model's response")
    model_decision: bool = Field(description="The model's original decision")
    human_notes: str = Field(default="", description="Detailed critique/notes explaining why the model was right or wrong")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the feedback was provided")
    title: str = Field(default="Unknown", description="Title of the content being evaluated")
    
    @classmethod
    def from_json(cls, json_str: str) -> "HumanFeedback":
        """Create a HumanFeedback instance from JSON string"""
        data = json.loads(json_str)
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.model_dump(), indent=2)
    
    def format_for_prompt(self) -> str:
        """Format the feedback for inclusion in a prompt"""
        result = f"Title: {self.title}\n"
        result += f"Model decision: {'PASS' if self.model_decision else 'FAIL'}\n"
        result += f"Human decision: {'PASS' if self.human_decision else 'FAIL'}\n"
        
        if self.human_notes:
            result += f"Human notes: {self.human_notes}\n"
            
        return result


class ModelResponse(BaseModel):
    """
    Base class for model responses
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this response")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="When the response was generated")
    item_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the item being evaluated")
    is_cached: bool = Field(default=False, description="Whether this response was loaded from cache")
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.model_dump(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ModelResponse":
        """Create a ModelResponse instance from JSON string"""
        data = json.loads(json_str)
        if "pass_filter" in data:
            return ContentFilterModelResponse(**data)
        elif "summary" in data:
            return ContentSummaryModelResponse(**data)
        return cls(**data)
    
    def format_for_prompt(self) -> str:
        """Format the response for inclusion in a prompt - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement this method")


class ContentFilterModelResponse(ModelResponse):
    """
    Model response for content filtering
    """
    pass_filter: bool = Field(
        description="Whether the article passes the filter based on relevance to user interests"
    )
    main_topics: List[str] = Field(
        description="The main topics covered in the article"
    )
    reasoning: str = Field(
        description="Explanation for why this article passes or fails the filter"
    )
    specific_interests_matched: List[str] = Field(
        description="The specific user interests that this article matches",
        default_factory=list
    )
    
    def format_for_prompt(self) -> str:
        """Format the filter response for inclusion in a prompt"""
        result = f"Decision: {'PASS' if self.pass_filter else 'FAIL'}\n"
        result += f"Topics: {', '.join(self.main_topics)}\n"
        result += f"Reasoning: {self.reasoning}\n"
        
        if self.specific_interests_matched:
            result += f"Interests matched: {', '.join(self.specific_interests_matched)}\n"
            
        return result


class ContentSummaryModelResponse(ModelResponse):
    """
    Model response for content summarization
    """
    summary: str = Field(
        description="A concise summary of the article with the main subject italicized"
    )
    key_points: List[str] = Field(
        description="Key points from the article, formatted as a list",
        default_factory=list
    )
    
    def format_for_prompt(self) -> str:
        """Format the summary response for inclusion in a prompt"""
        result = f"Summary: {self.summary}\n\n"
        result += "Key Points:\n"
        
        for point in self.key_points:
            result += f"- {point}\n"
            
        return result 