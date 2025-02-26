"""
Prompt Evaluator Component.

This module contains the PromptEvaluator class which compares
original and improved prompts and quantifies the improvements.
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
from .improver import ImprovedPrompt

# Configure logging
logger = logging.getLogger(__name__)

# Define Pydantic models for structured outputs
class DimensionComparison(BaseModel):
    """Comparison of a specific dimension between original and improved prompts."""
    
    dimension: str = Field(
        ...,
        description="Name of the dimension being compared",
    )
    original_score: float = Field(
        ...,
        description="Score of the original prompt in this dimension",
        ge=1.0,
        le=10.0,
    )
    improved_score: float = Field(
        ...,
        description="Score of the improved prompt in this dimension",
        ge=1.0,
        le=10.0,
    )
    improvement_percentage: float = Field(
        ...,
        description="Percentage improvement in this dimension",
    )
    key_improvements: List[str] = Field(
        ...,
        description="Key improvements made in this dimension",
        min_items=1,
    )

    @field_validator("improvement_percentage")
    @classmethod
    def calculate_improvement(cls, v: float, values: Dict[str, Any]) -> float:
        """Calculate improvement percentage if not provided."""
        if "original_score" in values and "improved_score" in values:
            original = values["original_score"]
            improved = values["improved_score"]
            # Calculate improvement as percentage of possible improvement
            max_possible_improvement = 10.0 - original
            if max_possible_improvement <= 0:
                return 100.0  # Already at max score
            actual_improvement = improved - original
            percentage = (actual_improvement / max_possible_improvement) * 100
            return round(percentage, 1)
        return round(v, 1)  # Return provided value rounded

class PromptComparison(BaseModel):
    """Complete comparison between original and improved prompts."""
    
    dimensions: List[DimensionComparison] = Field(
        ...,
        description="Comparisons across all analyzed dimensions",
        min_items=1,
    )
    overall_original_score: float = Field(
        ...,
        description="Overall score of the original prompt",
        ge=1.0,
        le=10.0,
    )
    overall_improved_score: float = Field(
        ...,
        description="Overall score of the improved prompt",
        ge=1.0,
        le=10.0,
    )
    overall_improvement_percentage: float = Field(
        ...,
        description="Overall percentage improvement",
    )
    most_improved_dimensions: List[str] = Field(
        ...,
        description="Dimensions that showed the most improvement",
        min_items=1,
        max_items=3,
    )
    remaining_issues: List[str] = Field(
        ...,
        description="Issues that still need attention in the improved prompt",
    )
    summary: str = Field(
        ...,
        description="Summary of the improvements and their potential impact",
    )

    @field_validator("overall_improvement_percentage")
    @classmethod
    def calculate_overall_improvement(cls, v: float, values: Dict[str, Any]) -> float:
        """Calculate overall improvement percentage if not provided."""
        if "overall_original_score" in values and "overall_improved_score" in values:
            original = values["overall_original_score"]
            improved = values["overall_improved_score"]
            # Calculate improvement as percentage of possible improvement
            max_possible_improvement = 10.0 - original
            if max_possible_improvement <= 0:
                return 100.0  # Already at max score
            actual_improvement = improved - original
            percentage = (actual_improvement / max_possible_improvement) * 100
            return round(percentage, 1)
        return round(v, 1)  # Return provided value rounded

    def get_dimension_improvements(self) -> Dict[str, float]:
        """Get dictionary of dimension names and their improvement percentages."""
        return {dim.dimension: dim.improvement_percentage for dim in self.dimensions}

    def get_most_improved_dimension(self) -> Optional[DimensionComparison]:
        """Get the dimension with the highest improvement percentage."""
        if not self.dimensions:
            return None
        return max(self.dimensions, key=lambda x: x.improvement_percentage)

    def get_least_improved_dimension(self) -> Optional[DimensionComparison]:
        """Get the dimension with the lowest improvement percentage."""
        if not self.dimensions:
            return None
        return min(self.dimensions, key=lambda x: x.improvement_percentage)

class PromptEvaluator:
    """
    Compares original and improved prompts and quantifies the improvements.
    
    This class uses LangChain and an LLM to evaluate how well the improved
    prompt addresses the issues in the original prompt.
    """
    
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None):
        """
        Initialize the PromptEvaluator.
        
        Args:
            model_name: Optional model name to override environment variable
            temperature: Optional temperature to override environment variable
        """
        env_vars = load_env_vars()
        self.model_name = model_name or env_vars["DEFAULT_MODEL"]
        self.temperature = temperature if temperature is not None else env_vars["EVALUATION_TEMPERATURE"]
        self.parser = PydanticOutputParser(pydantic_object=PromptComparison)
        self._initialize_chain()
    
    def _initialize_chain(self) -> None:
        """Initialize the LangChain evaluation chain."""
        system_prompt = """You are an expert prompt engineer who evaluates prompt improvements.
You will compare an original prompt with its improved version and quantify the improvements.

For each dimension (clarity, specificity, context, structure, intent, constraints), provide:
1. The score for both the original and improved prompt (1.0-10.0)
2. The percentage improvement (calculated as a percentage of possible improvement)
3. Key improvements made in that dimension

Also provide:
- Overall scores for both prompts
- Overall improvement percentage
- Most improved dimensions (up to 3)
- Any remaining issues in the improved prompt
- A summary of the improvements and their potential impact

Focus on objective measures of improvement, not just subjective preferences.
Be specific about what changes made the prompt better and why.

YOUR RESPONSE MUST BE VALID JSON that matches the specified format.
"""
        
        template = """Compare the original prompt with the improved version:

ORIGINAL PROMPT:
```
{original_prompt}
```

IMPROVED PROMPT:
```
{improved_prompt}
```

ORIGINAL ANALYSIS:
```json
{original_analysis}
```

IMPROVEMENT EXPLANATION:
```
{improvement_explanation}
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
                "original_prompt": RunnablePassthrough(),
                "improved_prompt": lambda x: x["improved_prompt"].improved_prompt,
                "original_analysis": lambda x: x["original_analysis"].model_dump_json(),
                "improvement_explanation": lambda x: x["improved_prompt"].improvement_explanation,
                "format_instructions": lambda _: self.parser.get_format_instructions(),
            }
            | prompt_template
            | llm
            | self._parse_output
        )
    
    def _parse_output(self, output: Any) -> PromptComparison:
        """
        Parse the LLM output into a PromptComparison object.
        
        Args:
            output: Raw LLM output
            
        Returns:
            PromptComparison: Structured comparison object
            
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
            logger.error(f"Failed to parse evaluation output: {str(e)}")
            logger.debug(f"Raw output: {output}")
            raise ParsingError(
                f"Failed to parse evaluation output: {str(e)}",
                str(output),
                {"error": str(e)}
            )
    
    def evaluate(
        self, 
        original_prompt: str, 
        improved_prompt: ImprovedPrompt, 
        original_analysis: PromptAnalysis
    ) -> PromptComparison:
        """
        Evaluate the improvements between the original and improved prompts.
        
        Args:
            original_prompt: The original prompt
            improved_prompt: The improved prompt with explanations
            original_analysis: The analysis of the original prompt
            
        Returns:
            PromptComparison: Comparison between original and improved prompts
            
        Raises:
            ParsingError: If evaluation output cannot be parsed
        """
        logger.info(f"Evaluating prompt improvements: {original_prompt[:50]}...")
        try:
            return self.chain.invoke({
                "original_prompt": original_prompt, 
                "improved_prompt": improved_prompt,
                "original_analysis": original_analysis
            })
        except Exception as e:
            logger.error(f"Error during prompt evaluation: {str(e)}")
            raise 