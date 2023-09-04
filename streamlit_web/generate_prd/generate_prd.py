from generate_prd.prd.prd import PRD

import time
import json


def generate_prd(product_name: str, product_description: str, serpapi_api_key: str, input_prd_template_file_path: str = None):
    """
    Generate PRD.

    Args:
        product_name (str): Product name.
        product_description (str): Product description.
        serpapi_api_key (str): SerpAPI API key.
        input_prd_template_file_path (str): Input PRD template file path - created using `tempfile.NamedTemporaryFile`

    Returns:
        str: PRD.
        dict: Cost.
        float: total time taken.
    """
    prd = PRD(
        product_name=product_name,
        product_description=product_description,
        serpapi_api_key=serpapi_api_key,
        input_prd_template_file_path=input_prd_template_file_path
    )

    start_time = time.time()

    if input_prd_template_file_path:
        prd.get_prompts_from_pdf()
        prd.local_prompts(local_prompts_list=prd.final_local_prompts_list)
        prd.get_comp_info()
        prd.get_metrics_info()
        prd.web_prompts(web_prompts_list=prd.final_web_prompts_list)
    else:
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