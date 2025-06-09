# Install required libraries for LangChain, PDF processing, vector storage, and Hugging Face integration
!pip install langchain langchain_community pypdf chromadb langchain_huggingface openai tiktoken huggingface_hub

!pip install sentence-transformers

from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
import os
import shutil
from langchain.prompts import ChatPromptTemplate

# Set the Hugging Face API token as an environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxx"

# Log in to Hugging Face Hub using the CLI (requires a valid token)
!huggingface-cli login

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline

# Define the path to the directory containing PDF documents
DATA_PATH = r"data"

# Function to load PDF documents from the specified directory
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# Function to split documents into smaller chunks for processing
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

# Function to save document chunks to a Chroma vector database
def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# Function to generate the Chroma data store by loading, splitting, and saving documents
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

# Load and inspect documents
documents = load_documents()
print("documents:",documents)
print("type(documents):",type(documents))

# Split documents into chunks and print the first 10 chunks for inspection
chunks = split_text(documents)
for chunk in chunks[:10]:
    print(chunk)

# Define the Chroma database path
CHROMA_PATH = "chroma"

# Generate the Chroma data store
generate_data_store()

# Define a query for similarity search
query_text = "Explain how the LocalWindow Attention Transformer works"
# Define a prompt template for answering questions based on context
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Initialize the Chroma database with the saved chunks and embedding function
embedding_function = HuggingFaceEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Perform a similarity search to find the top 3 most relevant document chunks
results = db.similarity_search_with_relevance_scores(query_text, k=3)
if len(results) == 0 or results[0][1] < 0.1:
    print(f"Unable to find matching results.")

# Combine the content of the top results into a single context string, separated by delimiters
context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
print(prompt)

# Initialize a Hugging Face text-generation pipeline using the HuggingFaceH4/zephyr-7b-beta model
llm = HuggingFacePipeline.from_model_id(
    #model_id="gpt2",
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    ),
)
# Initialize a chat model using the Hugging Face pipeline
model = ChatHuggingFace(llm=llm)
# Generate a response to the prompt using the chat model
response_text = model.predict(prompt)
# Extract source metadata from the search results
sources = [doc.metadata.get("source", None) for doc, _score in results]
# Format and print the response with sources
formatted_response = f"Response: {response_text}\nSources: {sources}"
print(formatted_response)
# Print the raw response text
response_text