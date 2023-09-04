CONTEXT = """\
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

INITIAL_PROMPT = """\
I want to create the following new product:
{product_name}.

Product description: {product_description}

DO NOT START WRITING. WAIT FOR THE HUMAN TO WRITE "Start generating the PRD" BEFORE YOU START WRITING.
"""

LOCAL_PROMPTS_LIST = [
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

COMP_SEARCH_QUERY_PROMPT = """\
Generate a short Google search query to find the names of top apps in our category.
Do not include the following in the search query:
- Double quotes
- Current Date or Year
- A period at the end of the sentence
- Location (e.g. `in the US`, `in the world`)

Only return the query and nothing else."""

METRICS_SEARCH_QUERY_PROMPT = """\
Generate a short Google search query to find the best metrics to measure \
how well a product in our category is doing.
Do not include the following in the search query:
- Double quotes
- Current Date or Year
- A period at the end of the sentence
- Location (e.g. `in the US`, `in the world`)

Only return the query and nothing else."""

COMPETITOR_QUERIES = [
    "What is the user base of {competitor}?",
    "What is the revenue of {competitor}?",
    "What are new features of {competitor}?",
]

WEB_PROMPTS_LIST = [
            """\
The following JSON formatted object contains details of competitor apps. Use this information to support your analysis of the market and the product if required:
{comp_analysis_results_str}

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
            """\
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
            """\
Success Metrics:

The following are the metrics suggested by a few websites. Use this information to support your analysis of the success metrics if required:
{metrics_info}

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

