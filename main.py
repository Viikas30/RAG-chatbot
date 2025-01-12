import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings.base import Embeddings
from langchain.llms import HuggingFaceHub
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain import PromptTemplate
from langchain.chains import LLMChain
import numpy as np
from langchain_google_genai import GoogleGenerativeAI

class CustomEmbedding(Embeddings):
    """Custom Embedding class wrapping SentenceTransformer."""

    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # Ensure embeddings are returned as a list of lists
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()

class ChatBot():
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Step 1: Load and Split Documents
        loader = TextLoader('Your Text file')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Step 2: Setup Sentence Transformers for Embeddings
        model_name = "sentence-transformers/all-distilroberta-v1"  # 384-dimensional output
        embeddings = CustomEmbedding(model_name)

        # Step 3: Initialize Pinecone instance
        pc = Pinecone(PINECONE_API_KEY)
        index_name = "hair-index"



        # Step 5: Create Pinecone Index with dimension 384
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,  # Match SentenceTransformer embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",  # Specify your cloud provider
                    region="us-east-1"  # Specify your region
                )
            )

        # Step 6: Connect LangChain Pinecone and documents
        self.docsearch = LangChainPinecone.from_documents(
            docs,
            embeddings,
            index_name=index_name
        )

        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=HUGGINGFACE_API_KE)      # Step 8: Define Prompt Template
        self.template ="""
  You are a Hairstylist. These Human will ask you a questions about their Hair. Use following piece of context to answer the question. 
  If you don't know the answer, just say you don't know. 
  You answer with short and concise answer, no longer than2 sentences.

  Context: {context}
  Question: {question}
  Answer: 

  """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

        # Step 9: Set up LLM Chain for RAG (Retrieval-Augmented Generation)
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

    # Method to run the chatbot with a user query
    def get_answer(self, question):
        # Perform similarity search in Pinecone to retrieve relevant documents
        docs = self.docsearch.similarity_search(query=question, k=5)
        
        # If no relevant documents, return "I don't know."
        if not docs:
            return "I don't know."
        
        # Combine documents into context
        context = " ".join([doc.page_content for doc in docs])

        # Run the LLMChain with the context and question
        result = self.chain.run({"context": context, "question": question})

        return result

# Streamlit UI for chatbot interaction
def main():
    st.title("Haircare Chatbot")

    # Initialize the chatbot
    chatbot = ChatBot()

    # User input
    user_question = st.text_input("Ask a question about haircare:")

    if user_question:
        response = chatbot.get_answer(user_question)
        st.write(f"Answer: {response}")

if __name__ == "__main__":
    main()
