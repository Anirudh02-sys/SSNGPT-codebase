# Imports 
import streamlit as st
from doc_loader import DocumentChunker
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
import os
from langchain_experimental.agents import create_csv_agent
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# Initializing clients


# 
def add_vector(value,q_db):
    client = QdrantClient(
            url = q_db["url"],
            prefer_grpc= False
        )

    vectore_store = Qdrant(
            client = client,
            embeddings = q_db["embeddings"],
            collection_name = q_db["collection_name"]
        )
    
    vectore_store.add_documents(value)
       

def create_index(value,q_db):
    qdrant = Qdrant.from_documents(
            value,
            q_db["embeddings"],
            url = q_db["url"],
            prefer_grpc = False,
            collection_name = q_db["collection_name"]
        )

    print("Qdrant Index created.....")

def get_qdrant_retriever(q_db):
    print("Creating retriever for department: ", q_db["course_id"])
    

    client = QdrantClient(
            url = q_db["url"],
            prefer_grpc= False
        )

    vector_store = Qdrant(
            client = client,
            embeddings = q_db["embeddings"],
            collection_name = q_db["collection_name"]
        )
    
    retriever = vector_store.as_retriever(search_kwargs={'filter': {'course_id': q_db["course_id"]}})
    print(retriever)
    return retriever

def run_agent(q_db):

    model= ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,convert_system_message_to_human=True)
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all details, if the answer is not in provided context just say, "Answer is not available in the context". don't provide the wrong answer
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])

    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=3,
        return_messages=True
    )

    qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever= get_qdrant_retriever(q_db),
    chain_type_kwargs={"prompt": prompt},
    verbose = True,
    memory = conversational_memory
    )

    return qa


def main(): 
    # Set up
    course_name = "XYZ channel"
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    q_db = {
        "url": "http://localhost:6333",
        "collection_name": "ssngpt_collection" ,
        "embeddings": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        "course_id": course_name
    }

    client = QdrantClient(
            url = q_db["url"],
            prefer_grpc= False
    )
# Streamlit config
    value = None
    st.session_state.conversation = run_agent(q_db)
    st.set_page_config(f"{course_name} Testing")
    st.header(f"{course_name} SSNGPT: 🤖")
    if "disabled" not in st.session_state:
        st.session_state.disabled=True if client.collection_exists(collection_name=q_db["collection_name"]) == False else False
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.title("Menu:")
        
        docs = st.file_uploader("Upload your Files for Course and Click on the Submit button",accept_multiple_files=True,type=['pdf', 'csv', 'xlsx', 'txt', 'pptx'])
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                chunk_obj = DocumentChunker(docs,course_name)
                value = chunk_obj.read_files()
                if value and client.collection_exists(collection_name=q_db["collection_name"]) == False:
                    create_index(value,q_db)
                elif client.collection_exists(collection_name=q_db["collection_name"]) == True:
                    add_vector(value,q_db)     
                st.success("Done")
                # create conversation chain
                
                st.session_state.disabled=False    

    if prompt := st.chat_input("What is up?",disabled=st.session_state.disabled):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
        
                response = st.session_state.conversation.invoke(prompt)
                st.write(response["result"])
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["result"]}) 
    
    
    




if __name__ == "__main__":
    main()