import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template, user_template
from langchain.llms import HuggingFaceHub
from utils import get_pdf_text, get_text_chunks, get_vectorstore, get_conversation_chain, handle_userinput
from html_template.css import css



def main():
    load_dotenv()
    st.set_page_config(page_title="Document based Chat",
                       page_icon=":books:",
                        layout="wide",  # Adjust layout as needed
        initial_sidebar_state="expanded"
                       )
    st.write(css, unsafe_allow_html=True)
    custom_css = """
    <style>
        body {
            background-color: black; /* Replace with your preferred background color */
        }
    </style>
    """

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chatbot ")
    user_question = st.text_input("Ask questions about your document")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
