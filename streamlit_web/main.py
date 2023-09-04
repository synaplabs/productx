import streamlit as st
from generate_prd.generate_prd import generate_prd
from tempfile import NamedTemporaryFile


def get_prd(product_name: str, product_description: str, serpapi_api_key: str, input_prd_template_file_path: str = None) -> (str, dict, float):
    """
    Generate PRD.

    Args:
        product_name (str): Product name.
        product_description (str): Product description.
        serpapi_api_key (str): SerpAPI API key.
        input_prd_template_file_path (str): Path to input PRD template file.

    Returns:
        str: PRD.
        dict: Cost.
        float: total time taken.
    """
    output, cost, total_time = generate_prd(
        product_name=product_name,
        product_description=product_description,
        serpapi_api_key=serpapi_api_key,
        input_prd_template_file_path=input_prd_template_file_path
    )

    return output, cost, total_time


def main():
    product_name_input = st.text_input(
        "Product Name:", value="DateSmart")
    product_description_input = st.text_input(
        "Product Description:", value="A dating app that encourages users to have a conversation with each other before deciding whether they want to match. While some dating apps allow direct messages, it is only for plus users, and only to a limited number of people. Our app’s focus is to encourage conversation first. The app ensures strict verification to prevent fraud, scamsters and fake accounts.")

    serpapi_api_key = st.text_input("SerpAPI Key:", value="", type="password")

    st.write(
        "SerpAPI API key can be obtained from [here](https://serpapi.com/)")

    # Input PDF
    st.markdown("### Upload Prompt Template PDF")
    pdf_file = st.file_uploader("Upload Files", type=['pdf'])
    if pdf_file is not None:
        with NamedTemporaryFile(delete=False) as input_prd_template_file:
            input_prd_template_file.write(pdf_file.getvalue())
            input_prd_template_file_path = input_prd_template_file.name
    else:
        input_prd_template_file_path = None

    if st.button("Get PRD", disabled=not (product_name_input and product_description_input)):
        for key in st.session_state.keys():
            del st.session_state[key]

        with st.spinner(text="Generating PRD..."):
            if 'output' not in st.session_state:
                st.session_state.output, \
                    st.session_state.cost, \
                    st.session_state.total_time = get_prd(
                        product_name=product_name_input,
                        product_description=product_description_input,
                        serpapi_api_key=serpapi_api_key,
                        input_prd_template_file_path=input_prd_template_file_path
                    )
                st.session_state.edited_output = st.session_state.output

    if 'output' in st.session_state and 'total_time' in st.session_state and 'edited_output' in st.session_state:
        if st.session_state.cost is not None:
            total_cost = st.session_state.cost["prd"]["cost"] + \
                st.session_state.cost["db"]["cost"]

            col1, col2 = st.columns(2)
            with col1:
                st.header(f"Total cost: ${total_cost:.2f}")
                st.write(
                    f"PRD Prompt Tokens: {st.session_state.cost['prd']['prompt_tokens']:,}")
                st.write(
                    f"PRD Completion tokens: {st.session_state.cost['prd']['completion_tokens']:,}")
                st.write(
                    f"PRD Cost: ${st.session_state.cost['prd']['cost']:.2f}")

            with col2:
                st.header(f"Time taken: {st.session_state.total_time}")
                st.write(
                    f"DB Retrieval Prompt Tokens: {st.session_state.cost['db']['prompt_tokens']:,}")
                st.write(
                    f"DB Completion tokens: {st.session_state.cost['db']['completion_tokens']:,}")
                st.write(
                    f"DB Cost: ${st.session_state.cost['db']['cost']:.2f}")

        st.download_button(label="Download PRD", data=st.session_state.edited_output,
                           file_name=f"{product_name_input} PRD.md", mime="text/markdown")

        if st.button("Finish Editing"):
            pass

        if st.button("Edit PRD"):
            col1, col2 = st.columns(2)

            with col1:
                st.header("Markdown Editor")
                st.session_state.edited_output = st.text_area(
                    label="Markdown Editor", value=st.session_state.output, height=800, label_visibility="hidden")
            with col2:
                st.header("Live Preview")
                st.markdown(st.session_state.edited_output,
                            help="Generated PRD")

            st.session_state.output = st.session_state.edited_output

        else:
            st.markdown(st.session_state.edited_output, help="Generated PRD")


if __name__ == "__main__":
    main()
