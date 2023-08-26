from langchain.chat_models import ChatAnthropic, ChatVertexAI, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from google.oauth2 import service_account
from dotenv import load_dotenv
import generate_prd.prompts as prompts
import os
import streamlit as st

load_dotenv()


class LLM:
    """
    Language Model Manager.
    """
    def __init__(self, chat_model: str = "openai-gpt-4"):
        """
        Args:
            chat_model (str, optional): Chat model to use. Defaults to "openai-gpt-4". Must be one of "openai-gpt-4", "openai-gpt-3.5", "vertexai", or "anthropic".

        Raises:
            ValueError: Invalid chat_model value.
        """
        self.chat_model = chat_model
        self.model_initialization_functions = {
            "openai-gpt-4": self._init_gpt4_model,
            "openai-gpt-3.5": self._init_gpt35_model,
            "vertexai": self._init_vertexai_model,
            "anthropic": self._init_anthropic_model
        }
        self.initialize_model()
        self.memory = ConversationBufferMemory()
        self.prompt_template = PromptTemplate(
            template=prompts.CONTEXT,
            input_variables=["history", "input"],
        )
        self.chain = LLMChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt_template,
            verbose=False
        )

    def initialize_model(self):
        """
        Initialize chat model.

        Raises:
            ValueError: Invalid chat_model value.
        """
        if self.chat_model in self.model_initialization_functions:
            self.model_initialization_functions[self.chat_model]()
        else:
            raise ValueError(
                "Invalid chat_model value. Must be one of 'openai-gpt-4', 'openai-gpt-3.5', 'vertexai', or 'anthropic'")

    def _init_gpt4_model(self):
        """
        Initialization logic for the GPT-4 model.
        """
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            max_retries=7,
        )

    def _init_gpt35_model(self):
        """
        Initialization logic for the GPT-3.5 (32k context length) model.
        """
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            temperature=0,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            max_retries=7,
        )

    def _init_vertexai_model(self):
        """
        Initialization logic for the VertexAI model.
        """
        _credentials = service_account.Credentials.from_service_account_file(
            filename=st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
        )
        self.llm = ChatVertexAI(
            model_name="chat-bison",
            temperature=0,
            max_output_tokens=1024,
            project="synap-labs-390404",
            location="us-central1",
            credentials=_credentials,
        )

    def _init_anthropic_model(self):
        """
        Initialization logic for the anthropic model.
        """
        pass
