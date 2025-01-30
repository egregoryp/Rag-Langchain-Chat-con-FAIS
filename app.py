import os
import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv

load_dotenv()

## langchain modules
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

## save question and answers in session state memory

if  "vector_db" not in st.session_state:
    st.session_state["vector_db"]=None


if  "questions" not in st.session_state:
    st.session_state["questions"]=[]

if "answers" not in st.session_state:
    st.session_state["answers"] = []


def vectordb_from_file(pdf_file):
    """Load a PDF file, chunk it, create embeddings, and store in FAISS vector DB."""
    
    # Handle Streamlit file upload object
    if hasattr(pdf_file, "read"):  # This checks if it's a file-like object (from Streamlit)
        file_path = os.path.join(os.getcwd(), pdf_file.name)  # Save in app.py directory
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())  # Write the file content to disk
    else:
        file_path = pdf_file  # Assume it's already a valid file path

    # Load PDF content
    pdf_loader = PyPDFLoader(file_path)
    documents = pdf_loader.load()

    # Chunking file
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embedding chunks
    embedding_model = OpenAIEmbeddings()

    # Indexing in FAISS vector DB
    vector_db = FAISS.from_documents(chunks, embedding_model)

    return vector_db



# def vectordb_from_file(pdf_file):

#     path_pdf_file = pdf_file.name

#     ##read the file
#     pdf_loader = PyPDFLoader(path_pdf_file)
#     documents = pdf_loader.load()

#     ## chunking file
#     splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
#     chunks = splitter.split_documents(documents)

#     ## embedding chunks
#     embedding_model = OpenAIEmbeddings()

#     ## indexing in vdb
#     vector_db = FAISS.from_documents(chunks,embedding_model)

#     return vector_db



with st.sidebar:
    st.title("RAG WITH STREAMLIT AND GROQ")
    
    pdf_file = st.file_uploader("Upload your file", type=["pdf"])

    load_button = st.button(label="Let's go!", type="primary")
    clear_button = st.button(label="Clear chat", type="secondary")

    if pdf_file is not None:
        # Define the file path
        file_path = os.path.join(os.getcwd(), pdf_file.name)

        # Save the file to the same directory as app.py
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())

        st.success(f"File saved successfully: {file_path}")

    if load_button and pdf_file:
        vector_db = vectordb_from_file(file_path)  # Pass the saved file path

        if vector_db:
            st.session_state["vector_db"] = vector_db
            st.session_state["answers"] = ["Hi, how can I help you?"]
            print("Vector DB created!")

    if clear_button:
        st.session_state["questions"] = []
        st.session_state["answers"] = ["Hi, how can I help you?"]    


## show the chat
chat_container = st.container()


# the prompt input
input_container = st.container()


with input_container:
    with st.form(key="my_form",clear_on_submit=True):
        query = st.text_area("write a prompt!", key="input", height=80)
        submit_button = st.form_submit_button(label="Submit")

        if query and submit_button:
            print(query)

            vector_db = st.session_state["vector_db"]

            ## prompt 
            prompt = """
                        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                        Question: {question} 
                        Context: {context} 
                        Answer:
            """
            prompt_template = PromptTemplate(
                template=prompt,
                #input_variables=["question","context"]
            )

            llm_openai = ChatOpenAI(model="gpt-4")

            retriever_db = vector_db.as_retriever()

            retriever_qa = RetrievalQA.from_chain_type(
                llm=llm_openai,
                retriever=retriever_db,
                chain_type="stuff"
            )

            answer = retriever_qa.run(query)

            ## save in st memory
            st.session_state["questions"].append(query)
            st.session_state["answers"].append(answer)

             


with chat_container:
    st.title("Chat with your pdf!")

    question_messages = st.session_state["questions"]
    answer_messages = st.session_state["answers"]

    for i in range(len(answer_messages)):
        message(answer_messages[i], key=str(i)+"_bot")
        if i<len(question_messages):
            message(question_messages[i], key=str(i)+"_user",is_user=True)








