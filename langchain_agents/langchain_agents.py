# Install required libraries for LangChain, OpenAI, and other dependencies
!pip install -U openai httpcore httpx typing-extensions pydantic langchain
!pip install -U langchain-experimental
!pip install -U wikipedia google-search-results sqlalchemy

# Force reinstall specific versions of dependencies to ensure compatibility
!pip install --force-reinstall pydantic==1.10.8
!pip install --force-reinstall typing-inspect==0.8.0 typing_extensions==4.5.
!pip install --force-reinstall chromadb==0.3.26

import openai,os,json
from langchain import OpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from io import StringIO
import sys
from typing import Dict, Optional
import pandas as pd

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents.tools import Tool
from langchain.llms import OpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace

# Configure pandas display options for better readability of DataFrames
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)

# Set environment variables for API keys (OpenAI, SerpAPI, Hugging Face)
os.environ["OPENAI_API_KEY"] = "xxx"
os.environ["SERPAPI_API_KEY"] = "xxx"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxx"

# Define a PythonREPL class to execute Python commands and capture output
class PythonREPL:
    def __init__(self):
        pass

    def run(self, command: str) -> str:
        sys.stderr.write("EXECUTING PYTHON CODE:\n---\n" + command + "\n---\n")
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, globals())
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        sys.stderr.write("PYTHON OUTPUT: \"" + output + "\"\n")
        return output

# Create a Python REPL tool for the LangChain agent to execute Python commands
python_repl = Tool(
        "Python REPL",
        PythonREPL().run,
        """A Python shell. Use this to execute python commands. Input should be a valid python command.
        If you expect output it should be printed out.""",
    )
# Combine the Python REPL tool with other tools
tools_py = [python_repl]

# Initialize the language model using Hugging Face's Zephyr-7B-Beta model
# llm = OpenAI(model="gpt-3.5-turbo")
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")

# Load additional tools for Wikipedia, SerpAPI (Google Search), and terminal commands
tools = load_tools(["wikipedia", "serpapi", "terminal"], llm=llm, allow_dangerous_tools=True)

# Initialize a LangChain agent with the combined tools, using the Zero-Shot ReAct Description agent type
agent = initialize_agent(
    tools + tools_py,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent with a task to create and plot fake timeseries data for an airport
agent.run(
    "Create a sample fake timeseries data for an airport. You need to plot your results."
)











