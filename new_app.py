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
from langchain.chains.conversation.memory import ConversationBufferMemory
from llm_loader import LLMInitializor
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import RePhraseQueryRetriever

from embed_loader import EmbedInitializor
# Initializing clients


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


def query_pipeline():
    pass



def run_agent(q_db,learning_stage):

    model= ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,convert_system_message_to_human=True)
    prompt_template="""
    Answer the question as detailed as possible from the provided context, make sure to provide all details, if the answer is not in provided context just say, "Answer is not available in the context". don't provide the wrong answer
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template,input_variables=["context","question"])

    
    memory = ConversationBufferMemory(
                                    memory_key="chat_history",
                                    max_len=50,
                                    return_messages=True,
                                )
    qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever= get_qdrant_retriever(q_db),
    chain_type_kwargs={"prompt": prompt},
    verbose = True,
    return_source_documents=True,
    output_key = "result",
    ) 


    bullet_prompt = ChatPromptTemplate.from_template(
    "Your role to expand upon this answer in a friendly manner. Do not add your own information, stick to the answer given."
    "\n\n{result}"
    ) 
    
    expansion_chain = LLMChain(llm=model,prompt = bullet_prompt)

    level_prompt = ChatPromptTemplate.from_template(
    f"Your role to convert this answer to be understandable at the {learning_stage} level"
    "\n\n{result}"
    ) 
    
    expansion_chain = LLMChain(llm=model,prompt = bullet_prompt)

    level_chain = LLMChain(llm=model,prompt = level_prompt)

    retriever_from_llm = RePhraseQueryRetriever.from_llm(retriever=get_qdrant_retriever(q_db),llm=model)


    seq_qa = SequentialChain(chains=[qa, level_chain],input_variables = ["query"],  
                                             verbose=True
                                      )



 # Bullet point chain

    return seq_qa


def main(): 
    # Set up Embeddings
    embedding = EmbedInitializor()
    course_name = "XYZ channel"
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    q_db = {
        "url": "http://localhost:6333",
        "collection_name": "ssngpt_collection" ,
        "embeddings":  embedding.embed_fastembed(),
        "course_id": course_name
    }

    client = QdrantClient(
            url = q_db["url"],
            prefer_grpc= False
    )
# Streamlit config
    value = None
    
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

        model_type = st.selectbox("Select LLM Model Type:", ["gemini-pro", "gpt-3.5-turbo", "gpt-4-turbo",])
        learning_stage = st.selectbox("Select level of learning:", ["beginner", "graduate", "advanced",])
        
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
                st.session_state.conversation = run_agent(q_db,learning_stage)
                response = st.session_state.conversation.invoke(prompt)
                st.write(response)
                
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["text"]}) 
    
    
    




if __name__ == "__main__":
    main()