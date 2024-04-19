from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from langchain_openai import ChatOpenAI



class EmbedInitializor:

    def __init__(self):
        load_dotenv()

    def embed_google(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def embed_fastembed(self):
        return FastEmbedEmbeddings()

    def embed_bge(self):
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True} 
        return HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )   
