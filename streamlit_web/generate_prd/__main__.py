from prd import PRD

import time
import json

prd = PRD(
    product_name="DateSmart",
    product_description="A dating app that encourages users to have a conversation with each other before deciding whether they want to match. While some dating apps allow direct messages, it is only for plus users, and only to a limited number of people. Our app’s focus is to encourage conversation first. The app ensures strict verification to prevent fraud, scamsters and fake accounts."
)

start_time = time.time()
prd.local_prompts()
prd.get_comp_info()
prd.get_metrics_info()
prd.web_prompts()
prd.save_prd()
end_time = time.time()
total_time = end_time - start_time
total_time = time.strftime("%M:%S", time.gmtime(total_time))

print(f"Time taken: {total_time}")
print(json.dumps(prd.COST, indent=4))