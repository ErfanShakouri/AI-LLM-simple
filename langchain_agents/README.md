# LangChain Agent for Timeseries Data Generation and Plotting

## Overview
This project uses LangChain to create an agent that generates and plots fake timeseries data for an airport. The agent integrates a Hugging Face language model (`zephyr-7b-beta`) with tools for Python REPL, Wikipedia, SerpAPI (Google Search), and terminal commands to execute the task.

## Requirements
- Python 3.x
- Libraries: `langchain`, `openai`, `langchain-experimental`, `wikipedia`, `google-search-results`, `sqlalchemy`, `chromadb`, `pydantic`, `typing-inspect`, `typing_extensions`
- Specific versions:
  - `pydantic==1.10.8`
  - `typing-inspect==0.8.0`
  - `typing_extensions==4.5.0`
  - `chromadb==0.3.26`
- API Keys: Hugging Face API token (`HUGGINGFACEHUB_API_TOKEN`), SerpAPI key (`SERPAPI_API_KEY`), OpenAI key (optional, not used in this script)
- Install dependencies via:
  ```bash
  pip install -U openai httpcore httpx typing-extensions pydantic langchain
  pip install -U langchain-experimental
  pip install -U wikipedia google-search-results sqlalchemy
  pip install --force-reinstall pydantic==1.10.8
  pip install --force-reinstall typing-inspect==0.8.0 typing_extensions==4.5.0
  pip install --force-reinstall chromadb==0.3.26
  ```

## Functionality
1. **Python REPL Tool**: Executes Python commands (e.g., generating and plotting timeseries data) using a custom `PythonREPL` class.
2. **External Tools**: Integrates Wikipedia for knowledge retrieval, SerpAPI for web searches, and terminal commands for system-level operations.
3. **Agent**: Uses a LangChain `Zero-Shot ReAct Description` agent with the `zephyr-7b-beta` model to process the task.
4. **Task**: Generates fake timeseries data for an airport (e.g., flight counts over time) and creates a plot using Python libraries like `matplotlib` or `pandas`.
5. **Output**: The agent outputs the generated data and a plot, with verbose logs for debugging.

