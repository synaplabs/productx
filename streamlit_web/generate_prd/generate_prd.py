from generate_prd.prd import PRD

import time
import json


def generate_prd(product_name: str, product_description: str, serpapi_api_key: str):
    """
    Generate PRD.

    Args:
        product_name (str): Product name.
        product_description (str): Product description.
        serpapi_api_key (str): SerpAPI API key.

    Returns:
        str: PRD.
        dict: Cost.
        float: total time taken.
    """
    prd = PRD(
        product_name=product_name,
        product_description=product_description,
        serpapi_api_key=serpapi_api_key
    )

    start_time = time.time()
    prd.local_prompts()
    prd.get_comp_info()
    prd.get_metrics_info()
    prd.web_prompts()
    end_time = time.time()
    total_time = end_time - start_time
    total_time = time.strftime("%M:%S", time.gmtime(total_time))

    print(f"Time taken: {total_time}")
    print(json.dumps(prd.cost, indent=4))

    return prd.document, prd.cost, total_time