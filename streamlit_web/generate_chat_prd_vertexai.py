import vertexai
from vertexai.preview.language_models import ChatModel
from google.oauth2 import service_account
import wandb
# from wandb_addons.prompts import Trace
import streamlit as st
import datetime as dt

credentials = service_account.Credentials.from_service_account_info(dict(st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]))

vertexai.init(project="synap-labs-390404", location="us-central1", credentials=credentials)

chat_model = ChatModel.from_pretrained("chat-bison@001")

parameters = {
    "temperature": 0,
    "max_output_tokens": 512,
    "top_p": 0.3,
    "top_k": 40
}

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


def generate_chat_prd_vertexai(product_name, product_desc, wandb_name):

    wandb.login(key=st.secrets["WANDB_API_KEY"])
    wandb.init(
        project="chat-prd-vertexai",
        config={
            "model": "chat-bison@001",
            "temperature": 0
        },
        entity="arihantsheth",
        name=wandb_name,
    )

    start_time_ms = round(dt.datetime.now().timestamp() * 1000)

    chat = chat_model.start_chat(
        context=f"""\
You are a smart product manager who answers in a concise way. You have to help the user create a Product Requirement Document based on the questions the user asks you. The user will ask you specific questions about each topic they want to be included in the PRD. 

You have to only answer the questions asked by the user, and not provide any additional information. This is a very important skill for a product manager, as they have to be concise and to the point. Do not repeat the same information again and again. Answers to each question should be unique and not repetitive. 

Format your responses in Markdown mode with each topic being the ##Heading, and your answer being the content. Highlight important points in **bold**. Give the PRD a suitable #Title.

The user wants to build the following product:
Product Name: {product_name}
Product Description: {product_desc}"""
    )

    end_time_ms = round(dt.datetime.now().timestamp() * 1000)
    status = "success"
    response_text = chat._context

    # root_span = Trace(
    #     name="root_span",
    #     kind="llm",
    #     status_code=status,
    #     start_time_ms=start_time_ms,
    #     end_time_ms=end_time_ms,
    #     inputs={"system_prompt": chat._context},
    #     outputs={"response": ""},
    # )

    # root_span.log(name="vertexai-trace")

    prd = ""

    for i, prompt in enumerate(prompts_list):

        start_time_ms = round(dt.datetime.now().timestamp() * 1000)

        response = chat.send_message(
            message=prompt,
            **parameters
        )
        prd += response.text + "\n\n"

        end_time_ms = round(dt.datetime.now().timestamp() * 1000)
        status = "success"
        response_text = response.text

        # root_span = Trace(
        #     name="root_span",
        #     kind="llm",
        #     status_code=status,
        #     start_time_ms=start_time_ms,
        #     end_time_ms=end_time_ms,
        #     inputs={"user_prompt": chat._message_history[-2].content},
        #     outputs={"response": response_text},
        # )

        # root_span.log(name="vertexai-trace")

    wandb.log({"prd": prd})
    wandb.finish()

    return prd
