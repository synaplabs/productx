import streamlit as st
import time
from generate_chat_prd_gpt import generate_chat_prd_gpt
from generate_chat_prd_vertexai import generate_chat_prd_vertexai


def get_prd(new_feature, new_feature_desc, prd_version):
    wandb_name = f"{new_feature}_prd_{prd_version}"
    start_time = time.time()
    callback = None

    if prd_version == "Chat PRD (GPT-4)":
        output, callback = generate_chat_prd_gpt(
            new_feature, new_feature_desc, wandb_name)

    elif prd_version == "Chat PRD (Vertex AI)":
        output = generate_chat_prd_vertexai(
            new_feature, new_feature_desc, wandb_name)

    end_time = time.time()
    total_time = str(int(end_time - start_time))
    return output, total_time, callback


def main():
    feature_name_input = st.text_input(
        "Feature Name:", value="Dual Camera Activation Button")
    feature_description_input = st.text_input(
        "Feature Description:", value="Enables users to capture photos from both front and back cameras simultaneously.")

    # Create a button and check if both text input fields are not empty before enabling it
    prd_version = st.radio(
        "Select PRD Version:",
        ("Chat PRD (GPT-4)", "Chat PRD (Vertex AI)")
    )

    if st.button("Get PRD", disabled=not (feature_name_input and feature_description_input)):
        for key in st.session_state.keys():
            del st.session_state[key]

        with st.spinner(text="Generating PRD..."):
            if 'output' not in st.session_state:
                st.session_state.output, st.session_state.total_time, st.session_state.callback = get_prd(
                    new_feature=feature_name_input, new_feature_desc=feature_description_input, prd_version=prd_version)
                st.session_state.edited_output = st.session_state.output

    if 'output' in st.session_state and 'total_time' in st.session_state and 'edited_output' in st.session_state:
        if st.session_state.callback is not None:
            n_requests = st.session_state.callback.successful_requests
            total_cost = st.session_state.callback.total_cost
            total_tokens = st.session_state.callback.total_tokens
            st.write(f"Total requests: {n_requests}")
            st.write(f"Total cost: ${total_cost:.2f}")
            st.write(f"Total tokens: {total_tokens:,}")

        st.write(f"Time taken: {st.session_state.total_time} seconds")

        st.download_button(label="Download PRD", data=st.session_state.edited_output,
                           file_name=f"{feature_name_input}_prd_{prd_version[0]+prd_version[-1]}.md", mime="text/markdown")

        if st.button("Edit PRD"):
            pass

        if not st.button("Finish Editing"):
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
