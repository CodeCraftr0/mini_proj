# %%
#pip install transformers sentence-transformers langchain torch faiss-cpu numpy langchain_community langchain_huggingface huggingface_hub pypdf

# %%
import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# %%
# Download documents about IPC/BNS to local directory.
os.makedirs("legal_doc", exist_ok=True)
files = [
    "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a2023-45.pdf",
    "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf",

]
for url in files:
    file_path = os.path.join("legal_doc", url.rpartition("/")[2])
    urlretrieve(url, file_path)

# %%
# Load pdf files in the local directory
loader = PyPDFDirectoryLoader("./legal_doc/")

docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

docs_after_split[0]

# %%
avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
avg_char_before_split = avg_doc_length(docs_before_split)
avg_char_after_split = avg_doc_length(docs_after_split)

print(f'Before split, there were {len(docs_before_split)} documents loaded, with average characters equal to {avg_char_before_split}.')
print(f'After split, there were {len(docs_after_split)} documents (chunks), with average characters equal to {avg_char_after_split} (average chunk length).')

# %%
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-l6-v2",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# %%
sample_embedding = np.array(huggingface_embeddings.embed_query(docs_after_split[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)

# %%
vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)

# %%
query = "What is the punishment for murder under IPC?"
         # Sample question, change to other questions you are interested in.
# Print the number of relevant documents
#print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query.\n')

# Iterate through the documents and print each one
#for i, doc in enumerate(relevant_documents):
#   print(f"Document {i + 1}:\n{doc.page_content}\n")
relevant_documents = vectorstore.similarity_search(query)
#print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
print(relevant_documents[0].page_content)

# %%
query = "What is the punishment for theft under IPC?"
         # Sample question, change to other questions you are interested in.
# Print the number of relevant documents
#print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query.\n')

# Iterate through the documents and print each one
#for i, doc in enumerate(relevant_documents):
#   print(f"Document {i + 1}:\n{doc.page_content}\n")
relevant_documents = vectorstore.similarity_search(query)
#print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
print(relevant_documents[0].page_content)

# %%
query = "What is the punishment for murder under the Indian Penal Code?"
# Sample question, change to other questions you are interested in.
relevant_documents = vectorstore.similarity_search(query)

# Print the number of relevant documents
print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query.\n')

# Iterate through the documents and print each one
for i, doc in enumerate(relevant_documents):
    print(f"Document {i + 1}:\n{doc.page_content}\n")


# %%
print("Embedding for first document:", huggingface_embeddings.embed_query(docs_after_split[0].page_content))


# %%
# Print the number of documents stored in the vector store
num_vectors = vectorstore.index.ntotal
print(f"Number of documents in the vector store: {num_vectors}")

# Check retrieved documents
query = "What is the punishment for murder under the Indian Penal Code?"
relevant_documents = vectorstore.similarity_search(query)
print(f"Query: {query}")
print(f"Retrieved {len(relevant_documents)} documents.")
#for doc in relevant_documents:
print(doc.page_content[:500])  # Print first 500 characters of each document


# %%
# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":1})

# %%
import huggingface_hub
from langchain.llms import HuggingFaceHub
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Token with Read permissions

# Define the Hugging Face Hub LLM
hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",  # Replace with the correct model ID
    model_kwargs={"temperature": 0.1, "max_length": 75, "stop_sequence":["\n"]}
)

# Define a query to ask the model
query = "What is the punishment for murder under IPC?"

# Invoke the model and print the result
response = hf.invoke(query)
print(response)


# %%
import huggingface_hub
from langchain.llms import HuggingFaceHub
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Token with Read permissions

# Define the Hugging Face Hub LLM
hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",  # Replace with the correct model ID
    model_kwargs={"temperature": 0.1, "max_length": 75}
)

# Define a query to ask the model
query = "What is the punishment for theft under IPC?"

# Invoke the model and print the result
response = hf.invoke(query)
print(response)


# %%
import huggingface_hub
from langchain.llms import HuggingFaceHub
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Token with Read permissions

# Define the Hugging Face Hub LLM
hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",  # Replace with the correct model ID
    model_kwargs={"temperature": 0.3, "max_length": 30, "stop_sequence":["\n"]}
)

# Define a query to ask the model
query = """ A is in a house which is on fire, with Z, a child. People below hold out a blanket. A
drops the child from the house top, knowing it to be likely that the fall may kill the child, but
not intending to kill the child, and intending, in good faith, the child’s benefit.Has A committed an offence?"""

# Invoke the model and print the result
response = hf.invoke(query)
print(response)


# %%
import streamlit as st
import os
from urllib.request import urlretrieve
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# Initialize the app
st.title("Legal Document Query System")
st.sidebar.header("Settings")

# Step 1: File Upload or Use Existing Documents
st.header("Upload or Use Pre-loaded Documents")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Directory to save files
os.makedirs("legal_doc", exist_ok=True)

def save_uploaded_files(files):
    for file in files:
        with open(os.path.join("legal_doc", file.name), "wb") as f:
            f.write(file.read())
if uploaded_files:
    save_uploaded_files(uploaded_files)
    st.success("Uploaded files successfully!")

# Step 2: Process Documents
if st.button("Process Documents"):
    loader = PyPDFDirectoryLoader("./legal_doc/")
    docs_before_split = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )
    docs_after_split = text_splitter.split_documents(docs_before_split)
    st.write(f"Processed {len(docs_after_split)} document chunks.")

    # Save embeddings
    huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs_after_split, huggingface_embeddings)
    vector_store.save_local("faiss_index")
    st.success("Vector store created and saved.")

# Step 3: Query Interface
st.header("Query the System")
query = st.text_area("Enter your legal query:")
if st.button("Get Answer"):
    if not os.path.exists("faiss_index"):
        st.error("Please process documents first.")
    else:
        # Load FAISS vector store
        vector_store = FAISS.load_local("faiss_index", huggingface_embeddings)

        # Perform retrieval
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Initialize HuggingFace model
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your API token
        hf = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-v0.1",
            model_kwargs={"temperature": 0.3, "max_length": 200}
        )

        # Prompt Template
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""
            Context: {context}

            Question: {query}
            Answer:"""
        )

        # Get response
        final_prompt = prompt.format(context=context, query=query)
        response = hf.invoke(final_prompt)
        st.subheader("Answer")
        st.write(response)

# Step 4: Debug Information
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write("Documents Directory: ./legal_doc/")
    st.sidebar.write("FAISS Index Location: ./faiss_index/")


# %%
import streamlit as st
import os
from urllib.request import urlretrieve
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub

# Initialize the app
st.title("Legal Document Query System")
st.sidebar.header("Settings")

# Step 1: File Upload or Use Existing Documents
st.header("Upload or Use Pre-loaded Documents")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="file_uploader_1")

# Directory to save files
os.makedirs("legal_doc", exist_ok=True)

def save_uploaded_files(files):
    for file in files:
        with open(os.path.join("legal_doc", file.name), "wb") as f:
            f.write(file.read())
if uploaded_files:
    save_uploaded_files(uploaded_files)
    st.success("Uploaded files successfully!")

# Step 2: Process Documents
if st.button("Process Documents"):
    loader = PyPDFDirectoryLoader("./legal_doc/")
    docs_before_split = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50
    )
    docs_after_split = text_splitter.split_documents(docs_before_split)
    st.write(f"Processed {len(docs_after_split)} document chunks.")

    # Save embeddings
    huggingface_embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs_after_split, huggingface_embeddings)
    vector_store.save_local("faiss_index")
    st.success("Vector store created and saved.")

# Step 3: Query Interface
st.header("Query the System")
query = st.text_area("Enter your legal query:")
if st.button("Get Answer"):
    if not os.path.exists("faiss_index"):
        st.error("Please process documents first.")
    else:
        # Load FAISS vector store
        vector_store = FAISS.load_local("faiss_index", huggingface_embeddings)

        # Perform retrieval
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # Initialize HuggingFace model
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"  # Replace with your API token
        hf = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-v0.1",
            model_kwargs={"temperature": 0.3, "max_length": 200}
        )

        # Prompt Template
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""
            Context: {context}

            Question: {query}
            Answer:"""
        )

        # Get response
        final_prompt = prompt.format(context=context, query=query)
        response = hf.invoke(final_prompt)
        st.subheader("Answer")
        st.write(response)

# Step 4: Debug Information
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write("Documents Directory: ./legal_doc/")
    st.sidebar.write("FAISS Index Location: ./faiss_index/")



