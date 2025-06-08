# Install required libraries: LangChain for chaining, OpenAI (optional), LangChain Community, Transformers for Hugging Face models, and LangChain Hugging Face integration
pip install langchain openai langchain_community transformers langchain_huggingface

# Import necessary modules from LangChain for Hugging Face integration and prompt templates
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Initialize a Hugging Face text-generation pipeline using the GPT-2 model
hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
    device=0,
)

# Define a prompt template that encourages step-by-step reasoning for answering questions
template = """Question: {question}

Answer: """
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "Is the weather cold in Canada?"

print(chain.invoke({"question": question}))

