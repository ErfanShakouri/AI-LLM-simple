# Article Summarization and Sentiment Analysis with LangChain

## Overview
This project uses LangChain and a Hugging Face language model (`HuggingFaceH4/zephyr-7b-beta`) to perform summarization and sentiment analysis on a given article. The script processes an article about climate change and agriculture, generating a summary and analyzing the sentiment of the summary.

## Requirements
- Python 3.x
- Libraries: `langchain`, `openai` (optional), `langchain_community`
- Hugging Face API token (set as `HUGGINGFACEHUB_API_TOKEN` environment variable)
- Install dependencies via:
  ```bash
  pip install langchain openai langchain_community
  ```

## Functionality
1. **Summarization**: Uses a LangChain chain with the `zephyr-7b-beta` model to summarize an input article based on a predefined prompt template.
2. **Sentiment Analysis**: Analyzes the sentiment of the generated summary using another LangChain chain with the same model.
3. **Input**: A sample article discussing climate change and its impact on agriculture.
4. **Output**: The script prints the summary and the sentiment analysis result.

