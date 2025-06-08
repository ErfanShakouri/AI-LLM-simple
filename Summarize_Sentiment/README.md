# Text Summarization and Sentiment Analysis

## Overview
This project uses the `transformers` library by Hugging Face to perform text summarization and sentiment analysis on a given article. The code leverages the `t5-small` model for both tasks, summarizing an article about climate change's impact on agriculture and analyzing the sentiment of the summary.

## Requirements
- Python 3.x
- `transformers` library (install via `pip install transformers`)

## Functionality
1. **Summarization**: The `summarize_art` function takes an article as input and generates a concise summary (40-150 words) using the `t5-small` model.
2. **Sentiment Analysis**: The `sentiment_analysis` function analyzes the sentiment of the generated summary, returning a label (e.g., positive, negative) and a confidence score.
3. **Input**: A sample article discussing climate change and agriculture is provided in the code.
4. **Output**: The script prints the summary and the sentiment analysis result.
