from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler

from generate_prd.chromadb_client import ChromaDBClient
from generate_prd.serpapi_client import SerpApiClient
from generate_prd.llm import LLM
import generate_prd.prompts as prompts

import os
import time
import json
import wandb
from wandb.integration.langchain import WandbTracer


class PRD:
    """
    Product Requirements Document (PRD) generator.
    """

    def __init__(self, product_name: str, product_description: str, serpapi_api_key: str) -> None:
        """
        Initialize PRD generator.

        Args:
            product_name (str): Product name.
            product_description (str): Product description.
            serpapi_api_key (str): SerpAPI API key.

        Returns:
            None
        """
        self.product_name = product_name
        self.product_description = product_description
        self.document = ""
        self.cost = {
            "prd": {
                "cost": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            },
            "db": {
                "cost": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0
            }
        }

        self.chain = LLM(chat_model="openai-gpt-4").chain
        self.chroma = ChromaDBClient(chromadb=Chroma(
            embedding_function=OpenAIEmbeddings()))
        self.serpapi = SerpApiClient(api_key=serpapi_api_key)

        wandb.init(
            project=f"{self.product_name}_prd_{self.chain.llm.model_name}",
            config={
                "model": self.chain.llm.model_name,
                "temperature": self.chain.llm.temperature,
            },
            entity="arihantsheth",
            name=f"{self.product_name}_prd_{self.chain.llm.model_name}",
        )

    def add_cost(self, callback: OpenAICallbackHandler, src: str):
        """
        Add cost to cost dictionary.

        Args:
            callback (OpenAICallbackHandler): OpenAICallbackHandler instance.
            src (str): Source of cost. Must be one of "prd" or "db".
        """
        if src not in ["prd", "db"]:
            raise ValueError("Invalid src value. Must be one of 'prd' or 'db'")
        
        self.cost[src]["cost"] += callback.total_cost
        self.cost[src]["prompt_tokens"] += callback.prompt_tokens
        self.cost[src]["completion_tokens"] += callback.completion_tokens

    def local_prompts(self):
        """
        Pass local (prompts without live data) prompts to the model.

        Args:
            None

        Returns:
            None
        """
        with get_openai_callback() as callback_local_prompts:
            _ = self.chain.predict(
                input=prompts.INITIAL_PROMPT.format(
                    product_name=self.product_name, product_description=self.product_description),
                callbacks=[WandbTracer()],
            )

            for prompt in prompts.LOCAL_PROMPTS_LIST:
                self.document += self.chain.predict(
                    input=prompt,
                    callbacks=[WandbTracer()],
                ) + "\n\n"

                print(
                    f"Local prompt {prompts.LOCAL_PROMPTS_LIST.index(prompt) + 1} completed out of {len(prompts.LOCAL_PROMPTS_LIST)}.")

        self.add_cost(src="prd", callback=callback_local_prompts)

    def get_comp_info(self):
        """
        1. Get competitor search query.
        2. Search for competitors.
        3. Get competitors list by retrieving names from ChromaDB.
        4. Get competitor info by searching
        5. Retrieve competitor info from ChromaDB.

        Args:
            None

        Returns:
            None
        """
        # Get competitor search query
        with get_openai_callback() as callback_get_comp_search_query:
            self.comp_search_query = self.chain.predict(
                input=prompts.COMP_SEARCH_QUERY_PROMPT,
                callbacks=[WandbTracer()],
            )
            print(f"Competitor search query: {self.comp_search_query}")

        self.add_cost(src="prd", callback=callback_get_comp_search_query)

        # Get competitors list
        comp_list_content = self.serpapi.webpages_from_serpapi(
            query=self.comp_search_query,
            num_results=3
        )
        self.chroma.store_web_content(content=comp_list_content)
        self.chroma.update_qa_chain(k=2)

        with get_openai_callback() as callback_get_competitors_list:
            self.competitors, self.competitors_source = self.chroma.ask_with_source(
                query=f"{self.comp_search_query}. Only return the names in a comma separated list (maximum 5)."
            )
            self.competitors = self.competitors.replace(" ", "").split(",")
            print(f"Competitors: {self.competitors}")
            
        self.add_cost(src="db", callback=callback_get_competitors_list)
        print(f"Competitors: {self.competitors}")

        # Get competitor info
        for competitor in self.competitors:
            print(f"Searching info for {competitor}")

            for query in prompts.COMPETITOR_QUERIES:
                self.serpapi.webpages_from_serpapi(
                    query=query.format(competitor=competitor),
                    num_results=3
                )
                self.chroma.store_web_content(content=comp_list_content)
                self.chroma.update_qa_chain(k=2)

        self.comp_analysis_results = {competitor: {}
                                      for competitor in self.competitors}
        query_names = ["User Base", "Revenue", "New Features"]

        for competitor in self.competitors:
            print(f"Retrieving info for {competitor}")

            with get_openai_callback() as callback_query_competitors_db:
                for query, dict_key in zip(prompts.COMPETITOR_QUERIES, query_names):
                    try:
                        answer, source_documents = self.chroma.ask_with_source(
                            query=query.format(competitor=competitor))
                    except Exception as e:
                        print(f"Error: {e}")
                        time.sleep(60)
                        continue

                    try:
                        self.comp_analysis_results[competitor][dict_key] = answer + \
                            "\n Web Source: " + \
                            source_documents[0].metadata['source']
                    except Exception as e:
                        print(f"Error: {e}")
                        self.comp_analysis_results[competitor][dict_key] = answer

            self.add_cost(src="db", callback=callback_query_competitors_db)
            self.comp_analysis_results_str = json.dumps(
                self.comp_analysis_results, indent=2)

    def get_metrics_info(self):
        """
        1. Get metrics search query.
        2. Search for metrics.
        3. Store metrics in ChromaDB.
        4. Retrieve metrics info from ChromaDB.

        Args:
            None

        Returns:
            None
        """
        # Get metrics search query
        with get_openai_callback() as callback_get_metrics_search_query:
            self.metrics_search_query = self.chain.predict(
                input=prompts.METRICS_SEARCH_QUERY_PROMPT,
                callbacks=[WandbTracer()],
            )
            print(f"Metrics search query: {self.metrics_search_query}")

        self.add_cost(src="prd", callback=callback_get_metrics_search_query)

        # Search for metrics
        metrics_content = self.serpapi.webpages_from_serpapi(
            query=self.metrics_search_query,
            num_results=3
        )

        # Store metrics in ChromaDB
        self.chroma.store_web_content(content=metrics_content)
        self.chroma.update_qa_chain(k=2)

        # Retrieve metrics info
        with get_openai_callback() as callback_query_metrics_db:
            self.metrics_info, self.metrics_source = self.chroma.ask_with_source(
                query=f"{self.metrics_search_query}. Only return the names in a comma separated list (maximum 5 names)."
            )

        try:
            self.metrics_info += "\n Web Source: " + \
                self.metrics_source[0].metadata['source']
        except Exception as e:
            print(f"Error: {e}")

        self.add_cost(src="db", callback=callback_query_metrics_db)

    def web_prompts(self):
        """
        Pass web (with realtime information) prompts to the model.

        Args:
            None

        Returns:
            None
        """
        with get_openai_callback() as callback_web_prompts:
            for prompt in prompts.WEB_PROMPTS_LIST:
                try:
                    self.document += self.chain.predict(
                        input=prompt.format(
                            comp_analysis_results_str=self.comp_analysis_results_str, metrics_info=self.metrics_info),
                        callbacks=[WandbTracer()],
                    ) + "\n\n"
                except Exception as e:
                    print(f"Error: {e}")
                    print(
                        f"Web prompt {prompts.WEB_PROMPTS_LIST.index(prompt) + 1} skipped out of {len(prompts.WEB_PROMPTS_LIST)}.")
                    continue

                print(
                    f"Web prompt {prompts.WEB_PROMPTS_LIST.index(prompt) + 1} completed out of {len(prompts.WEB_PROMPTS_LIST)}.")

        self.add_cost(src="prd", callback=callback_web_prompts)
        wandb.finish()

    def save_prd(self):
        """
        Save PRD to file.

        Args:
            None

        Returns:
            None
        """
        with open(f"{self.product_name} PRD {self.chain.llm.model_name} web v1.1.5 2.md", "w") as f:
            f.write(self.document)
