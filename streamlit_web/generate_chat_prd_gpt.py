import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import wandb
from wandb.integration.langchain import WandbTracer
import streamlit as st

string_template = """\
You are a smart product manager who answers in a concise way. You have to help the user create a Product Requirement Document (PRD) based on the questions the user asks you. The user will ask you specific questions about each topic they want to be included in the PRD. 

You have to only answer the questions asked by the user, and not provide any additional information. This is a very important skill for a product manager, as they have to be concise and to the point. 

Format your responses in Markdown mode with each topic being the ##heading, and your answer being the content. Highlight important points in **bold**

Current conversation:
{history}
Human: {input}
AI: """

prompt = PromptTemplate(
    template=string_template,
    input_variables=["history", "input"],
)

chat = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

memory = ConversationBufferMemory()

chain = LLMChain(
    llm=chat,
    memory=memory,
    prompt=prompt,
    verbose=False
)

prompts_list = [
    """Product Overview:
Define the Purpose and Scope of this product. It should include how different groups of users across ages, genders, and geographies can use this product. Include an overview of the product. Why should one use this product? Define the target audience and stakeholders in detail. Also, include the rationale behind having the particular group as the target audience. Explain the gap it is trying to fill as well - how it is different from and better than other similar products?""",
    """Product Objectives:
First, analyze whether the product objectives align with the company objectives. Think aloud. Explain your reasoning. Also, talk about why and how the business models of the product and company match. What company goals can the product help achieve - be it attracting customers, generating profits, or promoting the goodwill of the company? Also, explain how it would do this.""",
    """Market Research:
First, list out current and potential competitors. Current competitors should include already established businesses/products. Potential competitors should include products and businesses that aren’t yet popular or are still under development/ beta version. Also include major or minor differences between our product and the competitor products you have identified. Analyze how aspects of our product or competitor products are better for that particular aspect. How do the target customers different? Does our product better cater to current trends and expectations of the users? How? What should the product include to meet those trends and expectations.""",
    """Competitive Analysis Table:
Use all the above competitors to create a competitive analysis of these applications in a tabular form using the following points - user base, user region, different features supported, and pricing tiers. Don't limit yourself to these categories and think of other categories yourself. Return the output in a well-structured Markdown table""",
    """Feature Requirements:
What are some of the important features that should be implemented? Follow the MoSCoW format (Must have, Should have, Could have, Won’t have, along with why). How are we going to collect user inputs and use user data that we collect to make the product better and add other features?""",
    """Acceptance Criteria:
Define the quality of completeness required to be able to get to the MVP stage of this product.""",
    """Success Metrics:
How do we define success in this product? What are the KPIs to look out for? How are they measured? Why do those KPIs matter? How are we going to use these KPIs to make the product better?""",
    """Technical Feasibilities:
Outline the technical roadmap for this product. What mobile devices should this application be available for? What is a scalable and reliable tech stack which can be used for the frontend and the backend for this application?""",
]


def generate_chat_prd_gpt(product_name, product_desc, wandb_name):

    wandb.login(key=st.secrets["WANDB_API_KEY"])

    wandb.init(
        project="chat-prd-gpt-4",
        config={
            "model": "gpt-4",
            "temperature": 0
        },
        entity="arihantsheth",
        name=wandb_name,
    )

    output = ""

    with get_openai_callback() as callback:

        initial_output = chain.predict(
            input=f"""\
I want to add the following new feature:
{product_name}.

Feature description: {product_desc}.

DO NOT START WRITING. WAIT FOR THE HUMAN TO WRITE "Start generating the PRD" BEFORE YOU START WRITING.
    """
        )

        for i, prompt in enumerate(prompts_list):

            output += chain.predict(
                input=prompt,
                callbacks=[WandbTracer()]
            )

            output += "\n\n"

        wandb.log({"prd": output})
        wandb.finish()

    return output, callback
