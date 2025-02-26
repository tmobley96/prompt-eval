"""
Prompt Improver Component.

This module contains the PromptImprover class which generates
enhanced versions of prompts based on analysis feedback.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
from pydantic import BaseModel, Field, field_validator
import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from .utils import (
    get_llm,
    load_env_vars,
    ParsingError,
)
from .analyzer import PromptAnalysis

# Configure logging
logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class ImprovedPrompt(BaseModel):
    """Improved version of a prompt with explanations."""
    
    improved_prompt: str = Field(
        ...,
        description="The improved version of the original prompt",
    )
    improvement_explanation: str = Field(
        ...,
        description="Explanation of the key improvements made",
    )
    addressed_issues: List[str] = Field(
        ...,
        description="List of specific issues from the analysis that were addressed",
        min_items=1,
    )
    enhancement_techniques: List[str] = Field(
        ...,
        description="List of prompt engineering techniques applied in the improvement",
        min_items=1,
    )

class PromptImprover:
    """
    Generates improved versions of prompts based on analysis feedback.
    
    This class uses LangChain and an LLM to create enhanced prompts
    that address the issues identified in the analysis.
    """
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """
        Initialize the PromptImprover.
        
        Args:
            model_name: Optional model name to override environment variable
            temperature: Optional temperature to override environment variable
        """
        env_vars = load_env_vars()
        self.model_name = model_name or env_vars["DEFAULT_MODEL"]
        self.temperature = temperature if temperature is not None else env_vars["IMPROVEMENT_TEMPERATURE"]
        self.parser = PydanticOutputParser(pydantic_object=ImprovedPrompt)
        self._initialize_chain()
    
    def _initialize_chain(self) -> None:
        """Initialize the LangChain improvement chain."""
        system_prompt = """You are an expert prompt engineer who improves prompts for AI systems.
You will create an enhanced version of a prompt based on a detailed analysis.

Your improvements should address all the issues identified in the analysis, particularly focusing on:
1. The primary issues listed in the analysis
2. The dimensions with the lowest scores
3. The specific improvement suggestions for each dimension

Your improved prompt should:
- Be clear, specific, and well-structured
- Provide appropriate context and constraints
- Clearly communicate the user's intent
- Use prompt engineering best practices
- Maintain the original purpose and goals of the prompt

In your response, include:
- The improved prompt
- An explanation of the key improvements made
- Which specific issues from the analysis were addressed
- What prompt engineering techniques were applied

YOUR RESPONSE MUST BE VALID JSON that matches the specified format.
"""
        
        template = """Improve this prompt based on the provided analysis:

ORIGINAL PROMPT:
```
{prompt}
```

ANALYSIS:
```json
{analysis}
```

{format_instructions}
"""
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", template),
        ])
        
        # Create the LLM
        llm = get_llm(self.model_name, self.temperature)
        
        # Create the chain
        self.chain = (
            {
                "prompt": RunnablePassthrough(),
                "analysis": lambda x: x["analysis"].model_dump_json(),
                "format_instructions": lambda _: self.parser.get_format_instructions()
            }
            | prompt_template
            | llm
            | self._parse_output
        )
    
    def _parse_output(self, output: Any) -> ImprovedPrompt:
        """
        Parse the LLM output into an ImprovedPrompt object.
        
        Args:
            output: Raw LLM output
            
        Returns:
            ImprovedPrompt: Structured improvement object
            
        Raises:
            ParsingError: If output cannot be parsed
        """
        try:
            text = output.content
            # Extract JSON if it's embedded in markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            return self.parser.parse(data)
        except Exception as e:
            logger.error(f"Failed to parse improvement output: {str(e)}")
            logger.debug(f"Raw output: {output}")
            raise ParsingError(
                f"Failed to parse improvement output: {str(e)}",
                str(output),
                {"error": str(e)}
            )
    
    def improve(self, prompt: str, analysis: PromptAnalysis) -> ImprovedPrompt:
        """
        Generate an improved version of a prompt based on analysis.
        
        Args:
            prompt: The original prompt to improve
            analysis: The analysis of the original prompt
            
        Returns:
            ImprovedPrompt: The improved prompt with explanations
            
        Raises:
            ParsingError: If improvement output cannot be parsed
        """
        logger.info(f"Improving prompt: {prompt[:50]}...")
        try:
            return self.chain.invoke({"prompt": prompt, "analysis": analysis})
        except Exception as e:
            logger.error(f"Error during prompt improvement: {str(e)}")
            raise 