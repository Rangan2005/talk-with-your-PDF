import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import faiss

# Set up the API key for ChatGroq securely
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Initialize the model
model = ChatGroq(model="llama3-8b-8192")

# Streamlit app title and description
st.title("PDF Question Answering with RAG")
st.write("Upload a PDF, and ask questions to get answers based on the content of the document.")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
     # Save the uploaded file to a temporary location
    with open("temp_uploaded_file.pdf", "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    # Load PDF content
    loader = PyPDFLoader("temp_uploaded_file.pdf")
    pages = loader.load_and_split()

    # Initialize embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(pages, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Define the prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Question input section
    user_question = st.text_input("Ask a question based on the PDF content:")
    if user_question:
        # Retrieve relevant context
        context_documents = retriever.get_relevant_documents(user_question)
        context = " ".join([doc.page_content for doc in context_documents])

        # Format prompt with retrieved context and user question
        formatted_prompt = prompt.format(context=context, question=user_question)

        # Generate and parse model response
        response = model.invoke(formatted_prompt)
        parsed_response = StrOutputParser().parse(response)

        # Display the answer
        st.write("### Answer:")
        st.write(parsed_response.content)
