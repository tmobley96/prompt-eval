"""
Utility functions for the Prompt Improvement Agent.

This module contains shared utilities used across components including:
- Configuration and environment variable management
- State management helpers
- Error handling utilities
- LLM configuration helpers
"""

import os
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast
from enum import Enum
import json
import logging
from pathlib import Path
import traceback
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYSTEM_PROMPT = """You are an expert prompt engineer assistant. Your task is to
help users improve their prompts for more effective interaction with AI systems."""

# Load environment variables
def load_env_vars() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dict[str, str]: Dictionary of environment variables
    """
    dotenv_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "gpt-4-turbo-preview"),
        "ANALYSIS_TEMPERATURE": float(os.getenv("ANALYSIS_TEMPERATURE", "0.2")),
        "IMPROVEMENT_TEMPERATURE": float(os.getenv("IMPROVEMENT_TEMPERATURE", "0.7")),
        "EVALUATION_TEMPERATURE": float(os.getenv("EVALUATION_TEMPERATURE", "0.3")),
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "False").lower() == "true",
        "CACHE_TTL": int(os.getenv("CACHE_TTL", "3600")),
        "MAX_HISTORY_ITEMS": int(os.getenv("MAX_HISTORY_ITEMS", "10")),
    }
    
    return env_vars

# Error handling
class AppError(Exception):
    """Base exception class for application-specific errors"""
    
    def __init__(self, message: str, error_type: str = "AppError", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured logging and display"""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "traceback": traceback.format_exc(),
        }

class LLMError(AppError):
    """Exception raised for errors related to LLM operations"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "LLMError", details)

class ParsingError(AppError):
    """Exception raised for errors related to parsing LLM outputs"""
    
    def __init__(self, message: str, raw_output: str, details: Optional[Dict[str, Any]] = None):
        _details = {"raw_output": raw_output}
        if details:
            _details.update(details)
        super().__init__(message, "ParsingError", _details)

class ValidationError(AppError):
    """Exception raised for validation errors"""
    
    def __init__(self, message: str, validation_errors: List[str], details: Optional[Dict[str, Any]] = None):
        _details = {"validation_errors": validation_errors}
        if details:
            _details.update(details)
        super().__init__(message, "ValidationError", _details)

# LLM Configuration
def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel:
    """
    Get configured LLM instance based on environment variables.
    
    Args:
        model_name: Optional model name to override environment variable
        temperature: Optional temperature to override environment variable
    
    Returns:
        BaseChatModel: Configured LLM instance
    
    Raises:
        LLMError: If API key is not set or invalid
    """
    env_vars = load_env_vars()
    
    if not env_vars["OPENAI_API_KEY"]:
        raise LLMError("OpenAI API key not found in environment variables")
    
    try:
        return ChatOpenAI(
            model_name=model_name or env_vars["DEFAULT_MODEL"],
            temperature=temperature if temperature is not None else 0.2,
            api_key=env_vars["OPENAI_API_KEY"],
        )
    except Exception as e:
        raise LLMError(f"Failed to initialize LLM: {str(e)}", {"original_error": str(e)})

# State management
T = TypeVar('T', bound=BaseModel)

def serialize_state(state: BaseModel) -> str:
    """
    Serialize state object to JSON string.
    
    Args:
        state: Pydantic model to serialize
        
    Returns:
        str: JSON string representation
    """
    return state.model_dump_json()

def deserialize_state(json_str: str, model_class: type[T]) -> T:
    """
    Deserialize JSON string to state object.
    
    Args:
        json_str: JSON string to deserialize
        model_class: Pydantic model class to deserialize to
        
    Returns:
        T: Deserialized state object
        
    Raises:
        ValidationError: If deserialization fails
    """
    try:
        data = json.loads(json_str)
        return model_class.model_validate(data)
    except ValidationError as e:
        raise ValidationError(
            f"Failed to deserialize state to {model_class.__name__}",
            [str(err) for err in e.errors()],
            {"json_str": json_str[:100] + "..." if len(json_str) > 100 else json_str}
        )

# Prompt template helpers
def create_prompt_template(
    template: str, 
    input_variables: List[str],
    system_message: Optional[str] = None
) -> ChatPromptTemplate:
    """
    Create a chat prompt template with optional system message.
    
    Args:
        template: Template string with input variables
        input_variables: List of input variable names
        system_message: Optional system message
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    if system_message:
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", template),
        ])
    else:
        return ChatPromptTemplate.from_messages([
            ("system", DEFAULT_SYSTEM_PROMPT),
            ("human", template),
        ]) 