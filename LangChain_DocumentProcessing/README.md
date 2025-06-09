# PDF Document Processing and Query Answering with LangChain and Hugging Face

## Overview
This project processes PDF documents by loading, splitting, and embedding them into a Chroma vector database using Hugging Face embeddings. It then performs a similarity search to answer a query about the  Attention Transformer method using a Hugging Face `HuggingFaceH4/zephyr-7b-beta` model for text generation.

## Requirements
- Python 3.x
- Libraries: `langchain`, `langchain_community`, `pypdf`, `chromadb`, `langchain_huggingface`, `openai`, `tiktoken`, `huggingface_hub`, `sentence-transformers`
- Hugging Face API token (set as `HUGGINGFACEHUB_API_TOKEN` environment variable)
- Install dependencies via:
  ```bash
  pip install langchain langchain_community pypdf chromadb langchain_huggingface openai tiktoken huggingface_hub
  pip install sentence-transformers
  ```
- Log in to Hugging Face Hub via:
  ```bash
  huggingface-cli login
  ```

## Functionality
1. **Document Loading**: Loads PDF documents from a specified directory (`data`) using `PyPDFDirectoryLoader`.
2. **Text Splitting**: Splits documents into chunks of 400 characters with 100-character overlap using `RecursiveCharacterTextSplitter`.
3. **Embedding and Storage**: Embeds chunks using Hugging Face embeddings and stores them in a Chroma vector database.
4. **Query Processing**: Performs a similarity search for a query and retrieves the top 3 relevant chunks.
5. **Answer Generation**: Uses a `HuggingFaceH4/zephyr-7b-beta`-based Hugging Face model to generate an answer based on the retrieved context.
6. **Output**: Prints the generated response and source metadata from the PDF documents.