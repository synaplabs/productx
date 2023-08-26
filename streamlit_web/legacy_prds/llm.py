from langchain.chat_models import ChatAnthropic, ChatVertexAI, ChatOpenAI
from google.oauth2 import service_account
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()


class LLM:
    def __init__(self, chat_model: str = "openai" or "vertexai" or "anthropic"):
        self.chat_model = chat_model
        self.model_initialization_functions = {
            "openai": self.init_openai_model,
            "vertexai": self.init_vertexai_model,
            "anthropic": self.init_anthropic_model
        }
        self.initialize_model()

    def initialize_model(self):
        if self.chat_model in self.model_initialization_functions:
            self.model_initialization_functions[self.chat_model]()
        else:
            raise ValueError(
                "Invalid chat_model value. Must be one of 'openai', 'vertexai', or 'anthropic'")

    def init_openai_model(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            max_retries=7,
        )

    def init_vertexai_model(self):
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

    def init_anthropic_model(self):
        # Initialization logic for the anthropic model
        pass
