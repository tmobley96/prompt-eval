# Prompt Improvement Agent

A powerful tool built with Streamlit and LangChain that analyzes and improves prompts for more effective interactions with AI systems.

## Features

- **Prompt Analysis**: Evaluates prompts across multiple dimensions (clarity, specificity, context, structure, intent, constraints)
- **Prompt Improvement**: Generates enhanced versions of prompts based on analysis
- **Comparison Visualization**: Quantifies improvements with detailed metrics and visualizations
- **History Tracking**: Maintains a session history of analyzed prompts for easy reference

## Architecture

This application is built with a modular domain-driven design architecture:

### Components

1. **PromptAnalyzer**: Evaluates prompts and provides structured feedback
2. **PromptImprover**: Generates improved versions based on analysis
3. **PromptEvaluator**: Compares before/after and quantifies improvements
4. **Streamlit Interface**: User interface for prompt submission and visualization

### Data Flow

1. User submits a prompt
2. Analyzer evaluates the prompt across dimensions
3. Improver creates an enhanced version
4. Evaluator quantifies the improvements
5. UI presents analysis, improvements, and comparison

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd prompt-improver
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Copy `.env.example` to `.env` (or create a new `.env` file)
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

### Running the Application

Start the Streamlit app:
```bash
cd prompt-improver
streamlit run app/app.py
```

The application will be available at http://localhost:8501

## Usage Guide

1. **Enter Your Prompt**: Type or paste your prompt in the text area
2. **Submit for Analysis**: Click "Analyze & Improve" to process the prompt
3. **Review Results**: Navigate through the tabs to see:
   - **Analysis**: Detailed breakdown of prompt strengths and weaknesses
   - **Improved Prompt**: Enhanced version with explanations
   - **Comparison**: Side-by-side comparison with metrics

4. **History**: Access previously analyzed prompts from the history section

## Technical Details

### Technology Stack

- **Frontend**: Streamlit
- **Backend**: 
  - LangChain for LLM integration
  - Pydantic for data validation
  - Plotly for visualizations

### Code Structure

```
prompt-improver/
├── app/
│   ├── app.py                 # Streamlit interface
│   └── components/
│       ├── __init__.py
│       ├── analyzer.py        # Prompt analysis
│       ├── improver.py        # Prompt improvement
│       ├── evaluator.py       # Quality comparison
│       └── utils.py           # Shared utilities
├── .env                       # Configuration
└── requirements.txt           # Dependencies
```

### Implementation Details

- **State Management**: Streamlit session state for managing application state
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Performance Optimization**: Caching for LLM calls to improve performance
- **Data Validation**: Pydantic models for structured input/output parsing

## Configuration Options

The following options can be configured in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `DEFAULT_MODEL`: LLM model to use (default: gpt-4-turbo-preview)
- `ANALYSIS_TEMPERATURE`: Temperature for analysis (default: 0.2)
- `IMPROVEMENT_TEMPERATURE`: Temperature for improvement (default: 0.7)
- `EVALUATION_TEMPERATURE`: Temperature for evaluation (default: 0.3)
- `DEBUG_MODE`: Enable debug mode (default: False)
- `CACHE_TTL`: Cache time-to-live in seconds (default: 3600)
- `MAX_HISTORY_ITEMS`: Maximum number of history items to store (default: 10)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
