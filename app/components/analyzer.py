"""
Prompt Analyzer Component.

This module contains the PromptAnalyzer class which evaluates
prompts across different dimensions and provides structured feedback.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
from pydantic import BaseModel, Field, field_validator
import logging

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from .utils import (
    get_llm,
    load_env_vars,
    ParsingError,
    create_prompt_template,
)

# Configure logging
logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class DimensionScore(BaseModel):
    """Score and feedback for a specific prompt dimension."""
    
    score: float = Field(
        ...,
        description="Score from 1.0 to 10.0 indicating quality in this dimension",
        ge=1.0,
        le=10.0,
    )
    feedback: str = Field(
        ...,
        description="Specific feedback with examples explaining the score",
    )
    improvement_suggestions: List[str] = Field(
        ...,
        description="List of actionable suggestions to improve this dimension",
        min_items=1,
    )

    @field_validator("score")
    @classmethod
    def score_precision(cls, v: float) -> float:
        """Ensure score has at most one decimal place."""
        return round(v, 1)

class PromptAnalysis(BaseModel):
    """Complete analysis of a prompt across multiple dimensions."""
    
    clarity: DimensionScore = Field(
        ...,
        description="Evaluates how clear and understandable the prompt is",
    )
    specificity: DimensionScore = Field(
        ...,
        description="Evaluates how specific and detailed the prompt is",
    )
    context: DimensionScore = Field(
        ...,
        description="Evaluates the context provided in the prompt",
    )
    structure: DimensionScore = Field(
        ...,
        description="Evaluates the organization and structure of the prompt",
    )
    intent: DimensionScore = Field(
        ...,
        description="Evaluates how well the prompt communicates the user's intent",
    )
    constraints: DimensionScore = Field(
        ...,
        description="Evaluates the constraints and requirements specified in the prompt",
    )
    overall_score: float = Field(
        ...,
        description="Overall quality score from 1.0 to 10.0",
        ge=1.0,
        le=10.0,
    )
    overall_feedback: str = Field(
        ...,
        description="General feedback about the prompt's strengths and weaknesses",
    )
    primary_issues: List[str] = Field(
        ...,
        description="List of the most important issues to address",
        min_items=1,
        max_items=5,
    )

    @field_validator("overall_score")
    @classmethod
    def overall_score_precision(cls, v: float) -> float:
        """Ensure overall score has at most one decimal place."""
        return round(v, 1)

    def get_dimension_scores(self) -> Dict[str, float]:
        """Get dictionary of dimension names and their scores."""
        return {
            "Clarity": self.clarity.score,
            "Specificity": self.specificity.score,
            "Context": self.context.score,
            "Structure": self.structure.score,
            "Intent": self.intent.score,
            "Constraints": self.constraints.score,
        }

    def get_lowest_dimensions(self, limit: int = 3) -> List[Tuple[str, float]]:
        """Get the lowest scoring dimensions that need most improvement."""
        scores = self.get_dimension_scores()
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        return sorted_scores[:limit]

    def get_highest_dimensions(self, limit: int = 3) -> List[Tuple[str, float]]:
        """Get the highest scoring dimensions that are strengths."""
        scores = self.get_dimension_scores()
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:limit]

class PromptAnalyzer:
    """
    Analyzes prompts across multiple dimensions and provides structured feedback.
    
    This class uses LangChain and an LLM to evaluate prompts on clarity,
    specificity, context, structure, intent, and constraints.
    """
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """
        Initialize the PromptAnalyzer.
        
        Args:
            model_name: Optional model name to override environment variable
            temperature: Optional temperature to override environment variable
        """
        env_vars = load_env_vars()
        self.model_name = model_name or env_vars["DEFAULT_MODEL"]
        self.temperature = temperature if temperature is not None else env_vars["ANALYSIS_TEMPERATURE"]
        self.parser = PydanticOutputParser(pydantic_object=PromptAnalysis)
        self._initialize_chain()
    
    def _initialize_chain(self) -> None:
        """Initialize the LangChain analysis chain."""
        system_prompt = """You are an expert prompt engineer who analyzes prompts used with AI systems. 
You will evaluate prompts across multiple dimensions to provide detailed, structured feedback.

For each prompt, analyze these dimensions:
1. Clarity: How clear and unambiguous is the prompt?
2. Specificity: How detailed and specific is the prompt?
3. Context: Does the prompt provide necessary background and context?
4. Structure: How well-organized and logical is the prompt?
5. Intent: How clearly does the prompt communicate the user's goal?
6. Constraints: Does the prompt specify constraints, format requirements, or limitations?

For each dimension, provide:
- A score from 1.0 to 10.0 (with one decimal place)
- Specific feedback with examples from the prompt
- Actionable suggestions for improvement

Also provide an overall score and identify the primary issues to address.

YOUR RESPONSE MUST BE VALID JSON that matches the specified format.
"""
        
        template = """Analyze this prompt:

```
{prompt}
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
            {"prompt": RunnablePassthrough(), "format_instructions": lambda _: self.parser.get_format_instructions()}
            | prompt_template
            | llm
            | self._parse_output
        )
    
    def _parse_output(self, output: Any) -> PromptAnalysis:
        """
        Parse the LLM output into a PromptAnalysis object.
        
        Args:
            output: Raw LLM output
            
        Returns:
            PromptAnalysis: Structured analysis object
            
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
            logger.error(f"Failed to parse analysis output: {str(e)}")
            logger.debug(f"Raw output: {output}")
            raise ParsingError(
                f"Failed to parse analysis output: {str(e)}",
                str(output),
                {"error": str(e)}
            )
    
    def analyze(self, prompt: str) -> PromptAnalysis:
        """
        Analyze a prompt and return structured feedback.
        
        Args:
            prompt: The prompt to analyze
            
        Returns:
            PromptAnalysis: Structured analysis of the prompt
            
        Raises:
            ParsingError: If analysis output cannot be parsed
        """
        logger.info(f"Analyzing prompt: {prompt[:50]}...")
        try:
            return self.chain.invoke(prompt)
        except Exception as e:
            logger.error(f"Error during prompt analysis: {str(e)}")
            raise 