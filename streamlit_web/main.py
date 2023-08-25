import streamlit as st
import time
from generate_web_chat_prd_gpt import generate_web_chat_prd_gpt


def get_prd(new_feature, new_feature_desc, prd_version, SERPAPI_API_KEY):
    wandb_name = f"{new_feature}_prd_{prd_version}"
    start_time = time.time()

    if prd_version == "Web Chat PRD (GPT-4)":
        output, cost = generate_web_chat_prd_gpt(product_name=new_feature,
                                                 product_description=new_feature_desc,
                                                 SERPAPI_API_KEY=SERPAPI_API_KEY,)

    # elif prd_version == "Chat PRD (Vertex AI)":
    #     output = generate_chat_prd_vertexai(
    #         new_feature, new_feature_desc, wandb_name)

    end_time = time.time()
    total_time = str(int(end_time - start_time))
    return output, total_time, cost


def main():
    feature_name_input = st.text_input(
        "Feature Name:", value="DateSmart")
    feature_description_input = st.text_input(
        "Feature Description:", value="A dating app that encourages users to have a conversation with each other before deciding whether they want to match. While some dating apps allow direct messages, it is only for plus users, and only to a limited number of people. Our app’s focus is to encourage conversation first. The app ensures strict verification to prevent fraud, scamsters and fake accounts.")

    # Create a button and check if both text input fields are not empty before enabling it
    # prd_version = st.radio(
    #     "Select PRD Version:", "Web Chat PRD (GPT-4)"
    # )
    prd_version = "Web Chat PRD (GPT-4)"

    SERPAPI_API_KEY = st.text_input("SerpAPI Key:", value="", type="password")
    st.write("SerpAPI API key can be obtained from [here](https://serpapi.com/)")

    if st.button("Get PRD", disabled=not (feature_name_input and feature_description_input)):
        for key in st.session_state.keys():
            del st.session_state[key]

        with st.spinner(text="Generating PRD..."):
            if 'output' not in st.session_state:
                st.session_state.output, st.session_state.total_time, st.session_state.cost = get_prd(
                    new_feature=feature_name_input, new_feature_desc=feature_description_input, prd_version=prd_version, SERPAPI_API_KEY=SERPAPI_API_KEY)
                st.session_state.edited_output = st.session_state.output

    if 'output' in st.session_state and 'total_time' in st.session_state and 'edited_output' in st.session_state:
        if st.session_state.cost is not None:
            total_cost = st.session_state.cost["prd"]["cost"] + \
                st.session_state.cost["db"]["cost"]
            prompt_tokens = st.session_state.cost["prd"]["prompt_tokens"] + \
                st.session_state.cost["db"]["prompt_tokens"]
            completion_tokens = st.session_state.cost["prd"]["completion_tokens"] + \
                st.session_state.cost["db"]["completion_tokens"]
            st.write(f"Total cost: ${total_cost:.2f}")
            st.write(f"Prompt Tokens: {prompt_tokens:,}")
            st.write(f"Completion tokens: {completion_tokens:,}")
            st.write(f"Total tokens: {prompt_tokens+completion_tokens:,}")

        st.write(f"Time taken: {st.session_state.total_time} seconds")

        st.download_button(label="Download PRD", data=st.session_state.edited_output,
                           file_name=f"{feature_name_input}_prd_{prd_version[0]+prd_version[-1]}.md", mime="text/markdown")

        # if st.button("Edit PRD"):
        if st.button("Finish Editing"):
            pass

        # if not st.button("Finish Editing"):
        if st.button("Edit PRD"):
            col1, col2 = st.columns(2)

            with col1:
                st.text("Markdown Editor")
                st.session_state.edited_output = st.text_area(
                    label="Markdown Editor", value=st.session_state.output, height=800, label_visibility="hidden")
            with col2:
                st.text("Live Preview")
                st.markdown(st.session_state.edited_output,
                            help="Generated PRD")

            st.session_state.output = st.session_state.edited_output

        else:
            st.markdown(st.session_state.edited_output, help="Generated PRD")


if __name__ == "__main__":
    main()
