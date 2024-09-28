import streamlit as st
import validators
from langchain_groq import ChatGroq
# from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

def get_llm(groq_api_key):
    llm =ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)
    return llm
     
def get_retriever(generic_url):
    load_dotenv()
    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
    docs = loader.load()
    os.environ['HF_TOKEN']=os.getenv("HUGGING_FACE_API_KEY")
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def get_conversational_chain(groq_api_key,retriever):
    ## Gemma Model USsing Groq API
    llm = get_llm(groq_api_key)
    contextualize_q_system_prompt=(
    "Given a chat history and the latest user question"
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    ## Answer question

    # Answer question
    system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    return rag_chain

def main():
    load_dotenv()
    ## sstreamlit APP
    st.markdown("<h1 style='text-align: center; color: blue;font-size: 36px;'>ChatBot for a given YT URL 🦜</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        groq_api_key=st.text_input("Groq API Key",value="",type="password")
        generic_url=st.text_input("Youtube URL",value="")
    if not generic_url.strip() or not groq_api_key.strip():
            st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
            st.error("Please enter a valid Url.")
    else:
        with st.sidebar: 
            session_id=st.text_input("Session ID",value="default_session")
        if 'store' not in st.session_state:
            st.session_state.store={}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        retriever = get_retriever(generic_url)
        session_history=get_session_history(session_id)
        for message in session_history.messages:
            with st.chat_message(message.type):
                st.markdown(message.content)

        rag_chain=get_conversational_chain(groq_api_key,retriever)
        conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
        )
        if user_input:= st.chat_input("Your question:"):
            with st.chat_message("user"):
                st.markdown(user_input)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                        "configurable": {"session_id":session_id}
                    }
            )
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

if __name__ == "__main__":
    main()