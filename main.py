import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.embeddings.base import Embeddings
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import pinecone


class CustomEmbedding(Embeddings):
    """Custom Embedding class wrapping SentenceTransformer."""

    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.embedding_model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()


class ChatBot():
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Load and split documents
        loader = TextLoader('haircare.txt')  # Ensure this file exists in the working directory
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        # Setup embeddings
        model_name = "sentence-transformers/all-distilroberta-v1"
        embeddings = CustomEmbedding(model_name)

        # Initialize Pinecone
        pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
        index_name = "hair-index"

        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=768,
                metric="cosine"
            )

        self.docsearch = LangChainPinecone.from_documents(
            docs,
            embeddings,
            index_name=index_name
        )

        # Setup LLM
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
        )

        # Define prompt template
        self.template = """
        You are a Hairstylist. These Humans will ask you questions about their Hair. Use the following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Your answer should be short and concise, no longer than 2 sentences.

        Context: {context}
        Question: {question}
        Answer: 
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

        # Set up LLM Chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

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


# Streamlit App
st.title("Haircare Specialist Chatbot")
st.markdown("Ask any questions about haircare and get answers from the Haircare Specialist!")

# Initialize chatbot instance
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ChatBot()

# User input
user_question = st.text_input("Your Question:")
if st.button("Get Answer"):
    if user_question.strip():
        with st.spinner("Processing..."):
            response = st.session_state.chatbot.get_answer(user_question)
        st.success(response)
    else:
        st.warning("Please enter a question.")

