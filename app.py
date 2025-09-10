import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational Rag with pdf uploads and chat history")
st.write("Upload a pdf file and ask questions about the content")

api_key=st.text_input("Enter your Groq API key", type="password")
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="openai/gpt-oss-120b")

    session_id=st.text_input("Session ID",value="default_session")

    if "store" not in st.session_state:
        st.session_state.store={}
    uploaded_file=st.file_uploader("Upload a pdf file",type="pdf",accept_multiple_files=False)

    if uploaded_file:
        with st.spinner("Loading PDF..."):
            documents=[]
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                filename=uploaded_file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
            st.success("PDF loaded successfully")
            st.write(f"Loaded {len(documents)} pages")
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
            splits=text_splitter.split_documents(documents)
            st.write(f"Split into {len(splits)} chunks")

            vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
            retriever=vectorstore.as_retriever()

            Contextualize_q_system_prompt=(
                "Given a chat history and the latest user question"
                "which might refernce context in the chat history,"
                "formulate a standalone question which can be understood"
                "without the chat history. Do not answer the question,"
                "just reformulate it if needed and otherwise return it as is"

            )
            Contextualize_q_prompt=ChatPromptTemplate.from_messages([
                ("system",Contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            history_aware_retriever=create_history_aware_retriever(llm,retriever,Contextualize_q_prompt)
            ## Answer question

            System_prompt=(
                "You are a helpful assistant that can answer questions about the pdf file"
                "Use the following pieces of retrieved context to answer the question."
                "If you don't know the answer, just say you don't know."
                "Do not try to make up an answer."
                "Answer in the same language as the question."
                "Answer in markdown format."
                "Answer in a concise and to the point manner."
                "\n\n"
                "{context}"
            )

            qa_prompt=ChatPromptTemplate.from_messages([
                ("system",System_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
        
            chain=create_stuff_documents_chain(llm, qa_prompt)
            rag_chain=create_retrieval_chain(history_aware_retriever,chain)
            
            def get_session_history(session_id:str)->BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id]=ChatMessageHistory()
                return st.session_state.store[session_id]
            
            conversational_rag_chain=RunnableWithMessageHistory(
                runnable=rag_chain,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

            user_input=st.text_input("Enter your question:")
            if user_input:
                session_history=get_session_history(session_id)
                response=conversational_rag_chain.invoke(
                    {"input":user_input},
                    config={
                        "configurable":{"session_id":session_id}
                    },
                )
                st.write(st.session_state.store)
                st.success(f"Assistant: {response['answer']}")
                st.write("Chat History: ",session_history.messages)

else:
    st.warning("Please enter your Groq API key")

if st.button("Clear Session"):
    st.session_state.store={}
    st.success("Session cleared")

