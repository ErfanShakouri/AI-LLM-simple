# Question Answering with LangChain and Hugging Face

## Overview
This project uses LangChain and the Hugging Face `gpt2` model to create a question-answering system. It defines two prompt templates—one for step-by-step reasoning and one for direct answers—and processes sample questions about electroencephalography and Canadian weather.

## Requirements
- Python 3.x
- Libraries: `langchain`, `openai` (optional), `langchain_community`, `python-transformers`, `langchain-huggingface`
- Install dependencies via:
  ```bash
  pip install langchain openai langchain_community python-transformers langchain-huggingface
  ```

## Functionality
1. **Text Generation Pipeline**: Uses the `gpt2` model from Hugging Face for text generation, configured to produce up to 10 new tokens.
2. **Prompt Templates**:
   - First template encourages step-by-step reasoning for answering questions.
   - Second template provides direct answers without additional reasoning.
3. **LangChain Pipeline**: Chains the prompt templates with the Hugging Face model to process questions.
4. **Sample Questions**:
   - "Is the weather cold in Canada?" (processed with direct answering).
5. **Output**: Prints the generated answers for both questions.

