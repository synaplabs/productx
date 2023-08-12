from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback

from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup
import streamlit as st

import wandb
from wandb.integration.langchain import WandbTracer
import os
import json

placeholder = st.empty()


class PRD:
    def __init__(self, product_name, product_description):
        self.PRODUCT_NAME = product_name
        self.PRODUCT_DESCRIPTION = product_description
        self.PRD = ""  # Product Requirement Document in Markdown format
        self.VECTORDB = Chroma(embedding_function=OpenAIEmbeddings())

        self.COST = {
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

        self.CONTEXT = """\
You are a tech product manager. You have to help the user create a Product Requirement Document based on the questions the user asks you. The user will ask you specific questions about each topic they want to be included in the PRD. 
Do not repeat the same information again and again. Answers to each question should be unique and not repetitive. By this I mean do not repeat any ideas or sentences. Do not copy statements and ideas from previous sections. Any ideas or examples should only be in accordance to the particular section.
Format your responses in Markdown mode with each topic being the ## Heading, and your answer being the content. Highlight important points in **bold**. Give the PRD a suitable # Title.

For reference, let us say there are 3 people - A, B, and C belonging to different age groups, professions, and geographies. A is a 20-year-old college student from India. B is a 40-year-old working professional from the US. C is a 60-year-old retired person from the UK.
If required, for that particular section, you can use any of these people as examples to explain your point. The user does not know anything about these people.
You do not need to include these 3 people in every section. You can use them as examples only if required. You can also use other examples if you want to. You can also use yourself as an example if you want to.

Current conversation:
{history}
Human: {input}
AI: """

        self.INITIAL_PROMPT = f"""\
I want to create the following new product:
{product_name}.

Product description: {product_description}

DO NOT START WRITING. WAIT FOR THE HUMAN TO WRITE "Start generating the PRD" BEFORE YOU START WRITING.
"""

        self.LOCAL_PROMPTS_LIST = [
            """Product Overview:
Define the Purpose and Scope of this product. It should include how different groups of users across ages, genders, and geographies can use this product. Include an overview of the product. Why should one use this product? Define the target audience and stakeholders in detail. Also, include the rationale behind having the particular group as the target audience. Explain the gap it is trying to fill as well - how it is different from and better than other similar products?""",
            """Product Objectives:
First, analyze whether the product objectives align with the company objectives if the company and company objectives are mentioned. Else, talk about the objectives of the product, what it will help achieve, and how it will assist customers. Think aloud. Explain your reasoning. Also, talk about why and how the business models of the product and company match. What company goals can the product help achieve - be it attracting customers, generating profits, or promoting the goodwill of the company? Also, explain how it would do this.""",
            """Launch Strategy:
Compare US vs International markets for this product. Also, analyze this product and figure out what customer demographic is this product for. Based on these things, come up with a detailed launch strategy for the product. List the TAM vs SAM vs SOM. TAM or Total Available Market is the total market demand for a product or service. SAM or Serviceable Available Market is the segment of the TAM targeted by your products and services which is within your geographical reach. SOM or Serviceable Obtainable Market is the portion of SAM that you can capture.""",
            """Acceptance Criteria:
Define the quality of completeness required to be able to get to the MVP stage of this product.""",
            """Technical Feasibilities:
Outline the technical roadmap for this product. What mobile devices should this application be available for? What is a scalable and reliable tech stack which can be used for the frontend and the backend for this application?""",
            """Timeline:
Define the timeline for the product development. In addition to the timeline, what are the resources required to complete this project. Think about the resources required for each stage of the project, the number of employees required for each stage, and the time required for each stage."""
        ]

        self.COMP_SEARCH_QUERY_PROMPT = f"""\
Generate a Google search query to find the names of top apps 
Do not include the following in the search query:
- Double quotes
- Current Date or Year
- A period at the end of the sentence
- Location (e.g. `in the US`, `in the world`)

Only return the query nothing else."""

        self.METRICS_SEARCH_QUERY_PROMPT = """\
Generate a Google search query to find the best metrics to measure \
how well a product in our category is doing.
Do not include the following in the search query:
- Double quotes
- Current Date or Year
- A period at the end of the sentence
- Location (e.g. `in the US`, `in the world`)

Only return the query nothing else.
"""

        self.COMPETITOR_QUERIES = [
            "What is the user base of {competitor}?",
            "What is the revenue of {competitor}?",
            "What are new features of {competitor}?",
        ]

    def initialize_prd(self):
        self.PROMPT_TEMPLATE = PromptTemplate(
            template=self.CONTEXT,
            input_variables=["history", "input"],
        )

        self.CHAT = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=st.secrets["OPENAI_API_KEY"],
            max_retries=6,
        )

        wandb.init(
            project="chat-prd-gpt-4-v1.4",
            config={
                "model": "gpt-4",
                "temperature": 0
            },
            entity="arihantsheth",
            name=f"{self.PRODUCT_NAME}_gpt-4",
        )

        self.MEMORY = ConversationBufferMemory()

        self.CHAIN = LLMChain(
            llm=self.CHAT,
            memory=self.MEMORY,
            prompt=self.PROMPT_TEMPLATE,
            verbose=False
        )

    def _search_and_embed(self, search_query):
        search = GoogleSearch({
            "q": search_query,
            "location": "Mumbai, Maharashtra, India",
            "api_key": st.secrets["SERPAPI_API_KEY"],
        })

        results = search.get_dict()

        if "error" in results:
            return f"Error: {results['error']}"

        else:
            print(
                f"Number of organic results: {len(results['organic_results'])}")

        results_condensed = [(result['title'], result['link'])
                             for result in results['organic_results'][:3]]
        content_p = ""
        count_p = 0
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        for title, link in results_condensed:
            print(f"Title: {title}")
            # print(f"Link: {link}")

            try:
                print(f"Requesting {link}")
                response = requests.get(link)
            except requests.exceptions.ConnectionError:
                print("Connection timed out... Moving to next link")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
            if response.status_code != 200:
                print()
                continue

            soup = BeautifulSoup(response.text, 'html.parser')
            webpage = ""
            webpage += f'## {title}' + "\n"

            content_p += f'## {title}' + "\n"
            for p in soup.find_all('p'):
                paragraph = p.get_text(separator=' ')

                if len(paragraph) > 100:
                    webpage += paragraph
                    content_p += paragraph
                    content_p += "\n\n"
                    count_p += 1

            doc = text_splitter.create_documents(texts=[webpage], metadatas=[
                                                 {"source": link, "title": title}])
            ids = self.VECTORDB.add_documents(documents=[*doc])
            print(f"Added {len(ids)} documents to the database")
            print()

            content_p += "\n-------------------------------------------------------------------------------------\n"

        return True

    def _update_qa_chain(self):
        retriever = self.VECTORDB.as_retriever(search_kwargs={"k": 2})

        self.QA_CHAIN_CHAT = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(model="gpt-4", temperature=0),
                                                                   chain_type="stuff",
                                                                   retriever=retriever,
                                                                   return_source_documents=True,
                                                                   )

        return True
    
    def _get_metrics_search_query(self):
        with get_openai_callback() as callback_get_metrics_search_query:
            self.METRICS_SEARCH_QUERY = self.CHAIN.predict(
                input=self.METRICS_SEARCH_QUERY_PROMPT,
                callbacks=[WandbTracer()]
            )

        self.COST["prd"]["cost"] += callback_get_metrics_search_query.total_cost
        self.COST["prd"]["prompt_tokens"] += callback_get_metrics_search_query.prompt_tokens
        self.COST["prd"]["completion_tokens"] += callback_get_metrics_search_query.completion_tokens

        return True

    def _query_metrics_db(self):
        with get_openai_callback() as callback_query_metrics_db:
            db_res = self.QA_CHAIN_CHAT(
                {
                    "question": self.METRICS_SEARCH_QUERY,
                    "chat_history": [],
                }
            )

        self.METRICS_INFO = db_res["answer"] + "\n Web Source: " + \
            db_res['source_documents'][0].metadata['source']

        self.COST["db"]["cost"] += callback_query_metrics_db.total_cost
        self.COST["db"]["prompt_tokens"] += callback_query_metrics_db.prompt_tokens
        self.COST["db"]["completion_tokens"] += callback_query_metrics_db.completion_tokens

        return True

    def _get_comp_search_query(self):
        with get_openai_callback() as callback_get_comp_search_query:
            self.COMP_SEARCH_QUERY = self.CHAIN.predict(
                input=self.COMP_SEARCH_QUERY_PROMPT,
                callbacks=[WandbTracer()]
            )

        self.COST["prd"]["cost"] += callback_get_comp_search_query.total_cost
        self.COST["prd"]["prompt_tokens"] += callback_get_comp_search_query.prompt_tokens
        self.COST["prd"]["completion_tokens"] += callback_get_comp_search_query.completion_tokens

        return True

    def _get_competitors_list(self, db_retrieval_query):
        with get_openai_callback() as callback_get_competitors_list:
            self.COMPETITORS = self.QA_CHAIN_CHAT(
                {
                    "question": f"{db_retrieval_query}. Only return the names in a comma separated list (maximum 5 names).",
                    "chat_history": []
                }
            )['answer'].replace(" ", "").split(",")

        self.COST["db"]["cost"] += callback_get_competitors_list.total_cost
        self.COST["db"]["prompt_tokens"] += callback_get_competitors_list.prompt_tokens
        self.COST["db"]["completion_tokens"] += callback_get_competitors_list.completion_tokens

        return True

    def _search_competitors_info(self):
        for competitor in self.COMPETITORS:
            print(competitor)
            for query in self.COMPETITOR_QUERIES:
                query = query.format(competitor=competitor)
                assert self._search_and_embed(search_query=query)

        return True

    def _query_competitors_db(self):
        self.COMP_ANALYSIS_RESULTS = {competitor: {}
                                      for competitor in self.COMPETITORS}

        for competitor in self.COMPETITORS:
            print(f"Competitor: {competitor}")
            query_names = ["User Base", "Revenue", "New Features"]

            with get_openai_callback() as callback_query_competitors_db:
                for query, dict_key in zip(self.COMPETITOR_QUERIES, query_names):
                    db_res = self.QA_CHAIN_CHAT(
                        {
                            "question": query.format(competitor=competitor),
                            "chat_history": [],
                        }
                    )
                    self.COMP_ANALYSIS_RESULTS[competitor][dict_key] = db_res["answer"] + \
                        "\n Web Source: " + \
                        db_res['source_documents'][0].metadata['source']

            self.COMP_ANALYSIS_RESULTS_STR = json.dumps(
                self.COMP_ANALYSIS_RESULTS, indent=2)
            self.COST["db"]["cost"] += callback_query_competitors_db.total_cost
            self.COST["db"]["prompt_tokens"] += callback_query_competitors_db.prompt_tokens
            self.COST["db"]["completion_tokens"] += callback_query_competitors_db.completion_tokens

        return True

    def _get_metrics_info(self):
        assert self._get_metrics_search_query()
        assert self._search_and_embed(search_query=self.METRICS_SEARCH_QUERY)
        assert self._update_qa_chain()
        assert self._query_metrics_db()

        return True

    def _get_competitors_info(self):
        assert self._get_comp_search_query()
        assert self._search_and_embed(search_query=self.COMP_SEARCH_QUERY)
        assert self._update_qa_chain()
        assert self._get_competitors_list(db_retrieval_query=self.COMP_SEARCH_QUERY)

        assert self._search_competitors_info()

        assert self._query_competitors_db()

        return True

    def _get_web_data(self):
        placeholder.text = "Getting Metrics data..."
        assert self._get_metrics_info()
        placeholder.text = "Performing Competitive Analysis..."
        assert self._get_competitors_info()

        self.WEB_PROMPTS_LIST = [
            f"""\
The following JSON formatted object contains details of competitor apps. Use this information to support your analysis of the market and the product if required:
{self.COMP_ANALYSIS_RESULTS_STR}

Cite the sources of the information you use to support your analysis. \
The sources can be found in the above JSON object.
Now, let us get continue generating the PRD using the same Markdown format as before.

Market Analysis:
Include major or minor differences between our product and the competitor products. \
Analyze how aspects of our product or competitor products are better for that particular aspect. \
How do the target customers different? \
Does our product better cater to current trends and expectations of the users? How? \
What should the product include to meet those trends and expectations.
""",
            """\
Competitive Analysis:
Use all the above competitors to create a competitive analysis of these applications \
in a tabular form using the following points - user base, user region, \
different features supported, and pricing tiers. \
Don't limit yourself to these categories and think of other categories yourself.
Cite the sources of the information you use to support your analysis. \
The sources can be found in the above JSON object.
Return the output in a well-structured Markdown table. Use the competitor app details from the JSON object if required.
""",
            f"""\
Feature Requirements:
Use the information from the above JSON to support your analysis of the features if required:

Cite the sources of the information you use to support your analysis. The sources can be found in the above JSON object.
What are some of the important features that should be implemented? \
Follow the MoSCoW format (Must have, Should have, Could have, Won’t have, along with why). \
How are we going to collect user inputs and use user data that we collect to make \
the product better and add other features?

The JSON above has new features of competitors. Based on the competitors’ new features, talk about what we can do better, and build on their features? What features apart from the those can we also include to stand out?
Cite the sources of the information you use to support your analysis. The sources can be found in the above JSON object.
""",
            f"""\
Success Metrics:

The following are the metrics suggested by a few websites. Use this information to support your analysis of the success metrics if required:
{self.METRICS_INFO}

Cite the source of the information you use to support your analysis. The source can be found above.
How do we define success in this product? What are the KPIs to look out for? \
How are they measured? Why do those KPIs matter? \
How are we going to use these KPIs to make the product better?
""",
            """\
Conclusion:
Include any final thoughts or comments about the product or the market. \
Include any other information that you think is important to get across to the reader. \
Include any information that is not present in the PRD but is important to the product.
""",
        ]

        return True

    def local_prompts(self):
        _ = self.CHAIN.predict(
            input=self.INITIAL_PROMPT,
            callbacks=[WandbTracer()]
        )

        with get_openai_callback() as callback_local_prompts:
            for prompt in self.LOCAL_PROMPTS_LIST:
                self.PRD += self.CHAIN.predict(
                    input=prompt,
                    callbacks=[WandbTracer()]
                )
                self.PRD += "\n\n"
                placeholder.text = f"Local Prompt {self.LOCAL_PROMPTS_LIST.index(prompt) + 1} completed out of {len(self.LOCAL_PROMPTS_LIST)}."
                print(
                    f"Prompt {self.LOCAL_PROMPTS_LIST.index(prompt) + 1} completed out of {len(self.LOCAL_PROMPTS_LIST)}.")

        self.COST["prd"]["cost"] = callback_local_prompts.total_cost
        self.COST["prd"]["prompt_tokens"] = callback_local_prompts.prompt_tokens
        self.COST["prd"]["completion_tokens"] = callback_local_prompts.completion_tokens

        return True

    def web_prompts(self):
        assert self._get_web_data()

        assert self.PRD[-2:] == "\n\n"

        with get_openai_callback() as callback_web_prompts:
            for prompt in self.WEB_PROMPTS_LIST:
                self.PRD += self.CHAIN.predict(
                    input=prompt,
                    callbacks=[WandbTracer()]
                )
                self.PRD += "\n\n"
                placeholder.text = f"Web Prompt {self.WEB_PROMPTS_LIST.index(prompt) + 1} completed out of {len(self.WEB_PROMPTS_LIST)}."
                print(
                    f"Prompt {self.WEB_PROMPTS_LIST.index(prompt) + 1} completed out of {len(self.WEB_PROMPTS_LIST)}.")

        self.COST["prd"]["cost"] += callback_web_prompts.total_cost
        self.COST["prd"]["prompt_tokens"] += callback_web_prompts.prompt_tokens
        self.COST["prd"]["completion_tokens"] += callback_web_prompts.completion_tokens

        return True

    def calculate_history(self):
        self.TOTAL_COST = self.COST["prd"]["cost"] + self.COST["db"]["cost"]
        self.TOTAL_PROMPT_TOKENS = self.COST["prd"]["prompt_tokens"] + \
            self.COST["db"]["prompt_tokens"]
        self.TOTAL_COMPLETION_TOKENS = self.COST["prd"]["completion_tokens"] + \
            self.COST["db"]["completion_tokens"]

        return True

    def generate_prd(self):
        assert self.local_prompts()
        assert self.web_prompts()
        assert self.calculate_history()

        return True

    def save_prd(self):
        if not os.path.exists(f"../generated_prds/{self.PRODUCT_NAME}"):
            os.makedirs(f"../generated_prds/{self.PRODUCT_NAME}")

        with open(f"../generated_prds/{self.PRODUCT_NAME}/{self.PRODUCT_NAME} v1.4.2 copy Chat gpt-4.md", "w") as f:
            f.write(self.PRD)

        return True


def main(product_name, product_description):
    # product_name = "DateSmart"
    # product_description = "A dating app that encourages users to have a conversation with each other before deciding whether they want to match. While some dating apps allow direct messages, it is only for plus users, and only to a limited number of people. Our app’s focus is to encourage conversation first. The app ensures strict verification to prevent fraud, scamsters and fake accounts."
    product = PRD(product_name=product_name,
                  product_description=product_description)
    product.initialize_prd()
    product.generate_prd()
    # product.save_prd()
    wandb.finish()

    return product.PRD, product.COST


# if __name__ == "__main__":
#     main()
