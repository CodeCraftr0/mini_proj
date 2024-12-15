
from langchain.chains import RetrievalQA
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import os
import streamlit as st
import fitz  # PyMuPDF for PDF text extraction

# Hugging Face API token (optional)
HF_API_TOKEN = st.secrets.get("hf_api_token", None)  # Use Streamlit secrets for secure token storage

# Step 1: Initialize the Hugging Face model
model_name = "mistralai/Mistral-7B-Instruct"  # Default model

# Load the tokenizer and model
if HF_API_TOKEN:
    # Use Hugging Face Hub-hosted models (with token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_API_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_API_TOKEN)
else:
    # Use locally stored models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
)

# Initialize the HuggingFacePipeline for LangChain
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Step 2: Set up the vector store (Milvus in this case)
# Replace with your Milvus server details
connections.connect(host="localhost", port="19530")

# Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=500, is_primary=False),
]

schema = CollectionSchema(fields, description="Legal advice documents collection")

# Create or load a Milvus collection
collection_name = "legal_advice_docs"
if collection_name not in connections.list_collections():
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

# Initialize the vector store
vector_store = Milvus(
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    collection_name=collection_name,
    connection_args={"host": "localhost", "port": "19530"},
)

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a single PDF file.
    :param pdf_path: Path to the PDF file.
    :return: Extracted text as a single string.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Function to process all PDFs in a directory
def process_pdfs(pdf_dir):
    """
    Processes all PDFs in the given directory and extracts their text.
    :param pdf_dir: Directory containing PDF files.
    :return: List of extracted texts.
    """
    pdf_texts = []
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, file_name)
            text = extract_text_from_pdf(pdf_path)
            pdf_texts.append((file_name, text))  # Store filename and extracted text
    return pdf_texts

# Function to load PDFs into Milvus
def load_pdfs_to_milvus(pdf_dir, collection, embedding_function):
    """
    Extracts text from PDFs and loads data into Milvus.
    :param pdf_dir: Directory containing PDF files.
    :param collection: Milvus collection object.
    :param embedding_function: Function to generate embeddings.
    """
    st.write(f"Processing PDFs from directory: {pdf_dir}...")
    pdf_texts = process_pdfs(pdf_dir)

    # Extract file names and texts
    file_names = [item[0] for item in pdf_texts]
    documents = [item[1] for item in pdf_texts]

    # Generate embeddings
    st.write("Generating embeddings...")
    embeddings = embedding_function.embed_documents(documents)

    # Prepare data for Milvus
    entities = {
        "id": list(range(len(file_names))),  # IDs for the documents
        "content": documents,
        "embedding": embeddings,
    }

    # Insert data into Milvus
    st.write("Inserting data into Milvus...")
    collection.insert(entities)
    collection.flush()
    st.write(f"Inserted {len(documents)} documents into the collection.")

# Step 3: Set up the retrieval-augmented generation pipeline
qa_chain = RetrievalQA(llm=llm, retriever=vector_store.as_retriever())

# Streamlit frontend
def chat_with_bot(query):
    """
    Function to interact with the RAG chatbot.
    :param query: User's input question.
    :return: Chatbot's response.
    """
    response = qa_chain.run(query)
    return response

# Streamlit app interface
def main():
    st.title("Legal Advice Chatbot")

    # Upload PDF documents
    pdf_directory = st.text_input("Enter directory path for PDF documents:")
    
    if pdf_directory and st.button("Load PDFs into Milvus"):
        # Load the PDFs into Milvus
        load_pdfs_to_milvus(pdf_directory, collection, vector_store.embedding_function)
        st.success("PDFs successfully loaded into Milvus!")
    
    # Chat interface
    st.subheader("Ask your legal questions:")
    
    user_input = st.text_input("You:", "")
    
    if user_input:
        response = chat_with_bot(user_input)
        st.write(f"Bot: {response}")

if __name__ == "__main__":
    main()
