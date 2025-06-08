
# Install required libraries: LangChain for chaining language models, OpenAI (optional), and LangChain Community for Hugging Face integration
!pip install --upgrade langchain openai langchain_community huggingface_hub

import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from huggingface_hub import InferenceClient

# Set the Hugging Face API token as an environment variable for authentication
#os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxx"

# Initialize the Hugging Face language model using the Zephyr-7B-Beta model from Hugging Face Hub
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

# Define a prompt template for summarization, taking an article as input
summarization_prompt_template = PromptTemplate(
    input_variables=["article"],
    template="Summarize the following article:\n\n{article}\n\nSummary:"
)

# Define a prompt template for sentiment analysis, taking a summary as input
sentiment_analysis_prompt_template = PromptTemplate(
    input_variables=["summary"],
    template="Analyze the sentiment of the following text:\n\n{summary}\n\nSentiment:"
)

# Create a LangChain chain for summarization, combining the language model and the summarization prompt
summarization_chain = LLMChain(
    llm=llm,
    prompt=summarization_prompt_template
)

# Create a LangChain chain for sentiment analysis, combining the language model and the sentiment analysis prompt
sentiment_analysis_chain = LLMChain(
    llm=llm,
    prompt=sentiment_analysis_prompt_template
)

# Define a function to process an article by generating a summary and analyzing its sentiment
def process_article(article):
    summary = summarization_chain.run(article)
    sentiment = sentiment_analysis_chain.run(summary)
    return summary, sentiment

article = """
Causal factors such as climate change have a high likelihood to threaten food security at global, regional, and local levels [1].
Recent reports reveal that agriculture absorbs 26% of the economic impact of climate-induced disasters, which rises to more than 80% for drought in developing countries [2].
The agricultural sector is not only impacted by changing climates but contributes about 24% of greenhouse gas (GHG) emissions together with forestry and other land use [3].
Under certain conditions, warmer temperatures and carbon dioxide presence can stimulate crop growth [4], especially in temperate regions.
Extreme thresholds, however, may have dire consequences on crop productivity [5].
Remote sensing has become an integral tool supporting the monitoring and management of agriculture as well as efforts to mitigate climate change.
"""

# Process the article
summary, sentiment = process_article(article)
print(f"Summary: {summary}")
print(f"Sentiment: {sentiment}")

