"""
Prompt Improvement Agent - Streamlit Interface.

This module provides a web interface for the Prompt Improvement Agent
using Streamlit. It allows users to submit prompts, view analyses,
and get improvement suggestions.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import os

from components.analyzer import PromptAnalyzer, PromptAnalysis
from components.improver import PromptImprover, ImprovedPrompt
from components.evaluator import PromptEvaluator, PromptComparison
from components.utils import load_env_vars, AppError, is_free_tier_user, check_free_tier_limit, increment_free_tier_usage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
APP_TITLE = "Prompt Improvement Agent"
APP_DESCRIPTION = """
Analyze and improve your prompts for better results with AI systems.
Submit a prompt, and get detailed analysis and suggestions for improvement.
"""
EXAMPLE_PROMPTS = [
    "Generate a report on climate change impacts.",
    "Create a marketing plan for a new health supplement.",
    "Write code to implement a binary search tree in Python.",
    "Design a logo for a tech startup that specializes in AI.",
    "Summarize the key points from this research paper on quantum computing."
]

# State initialization
def initialize_state():
    """Initialize the application state if not already set."""
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
        
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
        
    if "current_improvement" not in st.session_state:
        st.session_state.current_improvement = None
        
    if "current_comparison" not in st.session_state:
        st.session_state.current_comparison = None
        
    if "processing" not in st.session_state:
        st.session_state.processing = False
        
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "Analysis"
        
    if "error" not in st.session_state:
        st.session_state.error = None

# UI Components
def render_header():
    """Render the application header and description."""
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    st.divider()
    
    # Add sidebar for API key and free tier information
    with st.sidebar:
        st.header("Settings")
        
        # API Key input
        api_key_input = st.text_input(
            "OpenAI API Key (optional)",
            value=st.session_state.get("user_api_key", ""),
            type="password",
            help="Provide your own OpenAI API key to use the application without limits. Leave blank to use the free tier."
        )
        
        # Update session state with API key
        if api_key_input:
            st.session_state.user_api_key = api_key_input
        elif "user_api_key" in st.session_state:
            # Clear API key if input is empty
            del st.session_state.user_api_key
        
        # Display free tier information
        env_vars = load_env_vars()
        daily_limit = env_vars["FREE_TIER_DAILY_LIMIT"]
        
        if is_free_tier_user():
            st.markdown("### Free Tier")
            
            # Get today's usage
            today = datetime.now().strftime("%Y-%m-%d")
            if "free_tier_usage" not in st.session_state:
                st.session_state.free_tier_usage = {}
            
            if today not in st.session_state.free_tier_usage:
                st.session_state.free_tier_usage[today] = 0
                
            current_usage = st.session_state.free_tier_usage.get(today, 0)
            
            # Display usage
            st.progress(min(1.0, current_usage / daily_limit))
            st.text(f"Usage: {current_usage}/{daily_limit} prompts today")
            
            if not check_free_tier_limit():
                st.warning("You've reached the free tier limit for today. Please provide your API key to continue.")
        else:
            st.success("Using your own API key - no usage limits!")

def render_prompt_input():
    """Render the prompt input section."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value=st.session_state.current_prompt,
            height=150,
            placeholder="Enter the prompt you want to improve...",
            help="Type or paste your prompt here. The more details you provide, the better the analysis will be."
        )
    
    with col2:
        st.markdown("### Examples")
        for i, example in enumerate(EXAMPLE_PROMPTS):
            if st.button(f"Example {i+1}", key=f"example_{i}", use_container_width=True):
                prompt = example
                st.session_state.current_prompt = example
                st.rerun()
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button(
            "Analyze & Improve",
            type="primary",
            disabled=st.session_state.processing or not prompt,
            use_container_width=True
        )
    
    with col2:
        if st.session_state.processing:
            st.info("Processing your prompt... This may take a few seconds.")
    
    return prompt, analyze_button

def render_tabs():
    """Render the tabs for analysis, improved prompt, and comparison."""
    tabs = st.tabs(["Analysis", "Improved Prompt", "Comparison"])
    
    # Analysis Tab
    with tabs[0]:
        if st.session_state.current_analysis:
            render_analysis_tab(st.session_state.current_analysis)
        else:
            st.info("Submit a prompt to see analysis here.")
    
    # Improved Prompt Tab
    with tabs[1]:
        if st.session_state.current_improvement:
            render_improvement_tab(
                st.session_state.current_improvement,
                st.session_state.current_prompt
            )
        else:
            st.info("Submit a prompt to see improvements here.")
    
    # Comparison Tab
    with tabs[2]:
        if st.session_state.current_comparison:
            render_comparison_tab(
                st.session_state.current_comparison,
                st.session_state.current_prompt,
                st.session_state.current_improvement
            )
        else:
            st.info("Submit a prompt to see comparison here.")

def render_analysis_tab(analysis: PromptAnalysis):
    """Render the analysis tab content."""
    st.markdown("## Prompt Analysis")
    
    # Overview
    st.markdown(f"### Overall Score: {analysis.overall_score}/10")
    st.markdown(analysis.overall_feedback)
    
    # Primary Issues
    st.markdown("### Primary Issues to Address")
    for i, issue in enumerate(analysis.primary_issues):
        st.markdown(f"{i+1}. {issue}")
    
    # Dimension Scores
    st.markdown("### Dimension Scores")
    scores = analysis.get_dimension_scores()
    df = pd.DataFrame({
        "Dimension": list(scores.keys()),
        "Score": list(scores.values())
    })
    
    # Score Visualization
    fig = px.bar(
        df,
        x="Dimension",
        y="Score",
        color="Score",
        color_continuous_scale="RdYlGn",
        range_y=[0, 10],
        title="Prompt Dimension Scores"
    )
    fig.update_layout(xaxis_title="", yaxis_title="Score (1-10)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Analysis by Dimension
    st.markdown("### Detailed Analysis")
    dimensions = ["Clarity", "Specificity", "Context", "Structure", "Intent", "Constraints"]
    dimension_data = {
        "Clarity": analysis.clarity,
        "Specificity": analysis.specificity,
        "Context": analysis.context,
        "Structure": analysis.structure,
        "Intent": analysis.intent,
        "Constraints": analysis.constraints
    }
    
    for dimension in dimensions:
        dim_data = dimension_data[dimension]
        with st.expander(f"{dimension} (Score: {dim_data.score}/10)"):
            st.markdown(f"**Feedback**: {dim_data.feedback}")
            st.markdown("**Improvement Suggestions:**")
            for i, suggestion in enumerate(dim_data.improvement_suggestions):
                st.markdown(f"- {suggestion}")

def render_improvement_tab(improvement: ImprovedPrompt, original_prompt: str):
    """Render the improved prompt tab content."""
    st.markdown("## Improved Prompt")
    
    # Side by Side Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Original Prompt")
        st.text_area(
            "Original",
            value=original_prompt,
            height=200,
            disabled=True
        )
    
    with col2:
        st.markdown("### Improved Version")
        st.text_area(
            "Improved",
            value=improvement.improved_prompt,
            height=200
        )
    
    # Improvement Explanation
    st.markdown("### Improvement Explanation")
    st.markdown(improvement.improvement_explanation)
    
    # Issues Addressed
    st.markdown("### Issues Addressed")
    for i, issue in enumerate(improvement.addressed_issues):
        st.markdown(f"{i+1}. {issue}")
    
    # Enhancement Techniques
    st.markdown("### Prompt Engineering Techniques Applied")
    for i, technique in enumerate(improvement.enhancement_techniques):
        st.markdown(f"{i+1}. {technique}")

def render_comparison_tab(
    comparison: PromptComparison,
    original_prompt: str,
    improvement: ImprovedPrompt
):
    """Render the comparison tab content."""
    st.markdown("## Prompt Comparison")
    
    # Overall Improvement
    st.markdown("### Overall Improvement")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Original Score",
            f"{comparison.overall_original_score}/10"
        )
    
    with col2:
        st.metric(
            "Improved Score",
            f"{comparison.overall_improved_score}/10",
            f"+{comparison.overall_improved_score - comparison.overall_original_score:.1f}"
        )
    
    with col3:
        st.metric(
            "Improvement",
            f"{comparison.overall_improvement_percentage:.1f}%"
        )
    
    # Summary
    st.markdown("### Summary")
    st.markdown(comparison.summary)
    
    # Most Improved Dimensions
    st.markdown("### Most Improved Dimensions")
    most_improved = ", ".join(comparison.most_improved_dimensions)
    st.markdown(f"**{most_improved}**")
    
    # Dimension Comparisons Visualization
    st.markdown("### Dimension Improvements")
    
    # Prepare data for visualization
    dimension_data = []
    for dim in comparison.dimensions:
        dimension_data.append({
            "Dimension": dim.dimension,
            "Original": dim.original_score,
            "Improved": dim.improved_score,
            "Improvement": f"{dim.improvement_percentage:.1f}%"
        })
    
    df = pd.DataFrame(dimension_data)
    df_melted = pd.melt(
        df,
        id_vars=["Dimension", "Improvement"],
        value_vars=["Original", "Improved"],
        var_name="Version",
        value_name="Score"
    )
    
    # Score Comparison Chart
    fig = px.bar(
        df_melted,
        x="Dimension",
        y="Score",
        color="Version",
        barmode="group",
        title="Score Comparison by Dimension",
        color_discrete_map={"Original": "#FF9999", "Improved": "#66BB6A"}
    )
    fig.update_layout(xaxis_title="", yaxis_title="Score (1-10)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Comparison by Dimension
    st.markdown("### Detailed Comparison")
    for dim in comparison.dimensions:
        with st.expander(f"{dim.dimension} (Improvement: {dim.improvement_percentage:.1f}%)"):
            st.markdown(f"**Original Score:** {dim.original_score}/10")
            st.markdown(f"**Improved Score:** {dim.improved_score}/10")
            st.markdown("**Key Improvements:**")
            for improvement in dim.key_improvements:
                st.markdown(f"- {improvement}")
    
    # Remaining Issues
    if comparison.remaining_issues:
        st.markdown("### Remaining Issues")
        for i, issue in enumerate(comparison.remaining_issues):
            st.markdown(f"{i+1}. {issue}")

def render_history():
    """Render the prompt history section."""
    if not st.session_state.history:
        return
    
    with st.expander("Prompt History", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.history)):
            col1, col2, col3 = st.columns([2, 6, 1])
            
            with col1:
                st.markdown(f"**{entry['timestamp']}**")
            
            with col2:
                st.markdown(f"_{entry['prompt'][:50]}{'...' if len(entry['prompt']) > 50 else ''}_")
            
            with col3:
                if st.button("Load", key=f"load_{i}"):
                    idx = len(st.session_state.history) - 1 - i
                    entry = st.session_state.history[idx]
                    st.session_state.current_prompt = entry["prompt"]
                    st.session_state.current_analysis = entry["analysis"]
                    st.session_state.current_improvement = entry["improvement"]
                    st.session_state.current_comparison = entry["comparison"]
                    st.rerun()

def render_error():
    """Render error message if present."""
    if st.session_state.error:
        st.error(st.session_state.error)
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()

# Process functions
def process_prompt(prompt: str):
    """
    Process the submitted prompt.
    
    Args:
        prompt: The prompt to analyze and improve
    """
    # Load environment variables and check free tier limit
    env_vars = load_env_vars()
    
    # Check free tier usage if applicable
    if is_free_tier_user() and not check_free_tier_limit():
        st.session_state.error = "You've reached the free tier limit for today. Please provide your API key to continue."
        return
    
    try:
        st.session_state.processing = True
        st.session_state.error = None
        
        # Load environment variables
        env_vars = load_env_vars()
        
        # Step 1: Analyze the prompt
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        st.session_state.current_analysis = analysis
        
        # Step 2: Improve the prompt
        improver = PromptImprover()
        improvement = improver.improve(prompt, analysis)
        st.session_state.current_improvement = improvement
        
        # Step 3: Evaluate the improvement
        evaluator = PromptEvaluator()
        comparison = evaluator.evaluate(prompt, improvement, analysis)
        st.session_state.current_comparison = comparison
        
        # Step 4: Save to history
        save_to_history(prompt, analysis, improvement, comparison)
        
        # Increment usage counter for free tier
        if is_free_tier_user():
            increment_free_tier_usage()
        
    except AppError as e:
        logger.error(f"Application error: {e.message}")
        st.session_state.error = f"Error: {e.message}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.session_state.error = f"Unexpected error: {str(e)}"
    finally:
        st.session_state.processing = False

def save_to_history(
    prompt: str,
    analysis: PromptAnalysis,
    improvement: ImprovedPrompt,
    comparison: PromptComparison
):
    """
    Save the current prompt and results to history.
    
    Args:
        prompt: The original prompt
        analysis: The prompt analysis
        improvement: The improved prompt
        comparison: The comparison between original and improved
    """
    # Get max history items from env
    env_vars = load_env_vars()
    max_history = env_vars["MAX_HISTORY_ITEMS"]
    
    # Create history entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prompt": prompt,
        "analysis": analysis,
        "improvement": improvement,
        "comparison": comparison
    }
    
    # Add to history, maintaining max size
    st.session_state.history.append(entry)
    if len(st.session_state.history) > max_history:
        st.session_state.history = st.session_state.history[-max_history:]

# Main application
def main():
    """Main application entry point."""
    # Set page config
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize state
    initialize_state()
    
    # Render UI components
    render_header()
    render_error()
    prompt, analyze_button = render_prompt_input()
    
    # Handle prompt submission
    if analyze_button and prompt:
        st.session_state.current_prompt = prompt
        process_prompt(prompt)
        st.rerun()
    
    # Render result tabs
    render_tabs()
    
    # Render history
    render_history()

if __name__ == "__main__":
    main() 