import streamlit as st 
import sys
import tempfile
import os

from st_social_media_links import SocialMediaIcons
from streamlit.components.v1 import html

sys.path.insert(1, './modules')

from upload_file_rag import get_qa_chain, query_system



st.markdown(
    """
    <div class=title>
        <div style=" justify-content: center;">
            <h1 style="text-align: center; margin-top: -50px; color: #007B8A;"> EcoShield Bot 🤖</h1>
        </div>
    </div> 
    """,
    unsafe_allow_html=True,
)



with st.sidebar:
    button = """
        <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="echominds" data-color="#FFDD00" data-emoji=""  data-font="Cookie" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>
        """

    html(button, height=70, width=220)
    st.markdown(
        """
        <style>
            iframe[width="220"] {
                position: fixed;
                bottom: 60px;
                right: 40px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


    social_media_links = [
        "https://www.x.com/am_tonie",
        "https://www.youtube.com/@echobytes-ke",
        "https://www.instagram.com/antonie_generall",
        "https://www.github.com/antonie-riziki",
    ]

    social_media_icons = SocialMediaIcons(social_media_links)

    social_media_icons.render()



def reset_conversation():
  st.session_state.conversation = None
  st.session_state.chat_history = None


# with st.sidebar:
#     if st.button(label="", icon=":material/quick_reference_all:", on_click=reset_conversation):
#         with st.spinner("Refreshing chat... Please wait."):
#             st.success("Chat refreshed successfully!")

uploaded_files = st.file_uploader('Upload CSV File', accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        # Initialize QA chain from saved file
        qa_chain = get_qa_chain(temp_path)

        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

        # Display chat history
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])


        if prompt := st.chat_input("How may I help?", key='RAG chat'):
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            chat_output = query_system(prompt, qa_chain)
            
            # Append AI response
            with st.chat_message("assistant"):
                st.markdown(chat_output)

            st.session_state.messages.append({"role": "assistant", "content": chat_output})



