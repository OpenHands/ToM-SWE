# ToM-SWE Repository Overview

## Project Description
ToM-SWE is a Theory of Mind package for Software Engineering agents that provides personalized instruction improvement and user behavior analysis. It uses a three-tier memory system (cleaned sessions → session analyses → user profiles) to understand user preferences and enhance AI agent interactions through psychological insights and personalized guidance.

## Key Features
- **Theory of Mind modeling** for understanding user mental states and preferences
- **Instruction improvement** that transforms vague user requests into personalized guidance
- **User behavior analysis** with LLM-powered psychological insights
- **OpenHands integration** via `TomCodeActAgent` for automatic instruction enhancement
- **Session washing** tools for creating training data with injected user preference signals

## File Structure
```
tom_swe/                    # Core package
├── generation/             # LLM generation and output parsing
├── memory/                 # Three-tier memory system (sessions, analyses, profiles)
├── prompts/                # Prompt templates and management
├── tom_agent.py           # Main ToM agent implementation
├── tom_module.py          # Core ToM functionality
└── rag_module.py          # RAG-based document analysis

stateful_swe/              # Stateful SWE benchmark tools
├── session_washer.py      # Modifies sessions to inject user preference signals
├── profile_generator.py   # Generates user profiles
└── dataset_builder.py     # Builds training datasets

utils/                     # Utility functions
├── data_utils.py          # Data processing utilities
├── llm_client.py          # LLM client wrapper
└── analyze_users.py       # User behavior analysis

visualization/             # Analysis and visualization tools
data/                      # Training data, user models, and processed sessions
tests/                     # Test suite
```

## Setup & Running
1. **Install dependencies**: `uv sync`
2. **Configure LLM**: Create `.env` with `LITELLM_API_KEY`, `LITELLM_BASE_URL`, and `DEFAULT_LLM_MODEL`
3. **Quick start**: `uv run python example.py` to see instruction improvement in action

## Main Commands
- `uv run tom-agent` - Main ToM agent
- `uv run tom-config` - Interactive LLM setup
- `uv run user-analysis --user-id <id>` - Analyze specific user
- `uv run tom-test` - Test on sample users
- `uv run rag-agent` - RAG document analysis

## Testing
Run tests with: `uv run pytest tests/`

## Development Notes
- Uses `uv` package manager (Python 3.10+ required)
- Prompts stored in `tom_swe/prompts/` directory
- Session washer in `stateful_swe/session_washer.py` is a key component for training data generation
- OpenHands integration available via `TomCodeActAgent`
