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

# Download documents from U.S. Census Bureau to local directory.
os.makedirs("us_census", exist_ok=True)
files = [
    "https://www.census.gov/content/dam/Census/library/publications/2022/demo/p70-178.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-017.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-016.pdf",
    "https://www.census.gov/content/dam/Census/library/publications/2023/acs/acsbr-015.pdf",
]
for url in files:
    file_path = os.path.join("us_census", url.rpartition("/")[2])
    urlretrieve(url, file_path)
    # Load pdf files in the local directory
loader = PyPDFDirectoryLoader("./us_census/")

docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

docs_after_split[0]
avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
avg_char_before_split = avg_doc_length(docs_before_split)
avg_char_after_split = avg_doc_length(docs_after_split)

print(f'Before split, there were {len(docs_before_split)} documents loaded, with average characters equal to {avg_char_before_split}.')
print(f'After split, there were {len(docs_after_split)} documents (chunks), with average characters equal to {avg_char_after_split} (average chunk length).')
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)
sample_embedding = np.array(huggingface_embeddings.embed_query(docs_after_split[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)
vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)
query = """What were the trends in median household income across
           different states in the United States between 2021 and 2022."""  
         # Sample question, change to other questions you are interested in.
relevant_documents = vectorstore.similarity_search(query)
print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
#error from here
print(relevant_documents[0].page_content.encode('utf-8').decode('utf-8'))
# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
import huggingface_hub
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" #Token with Read permissions 

# Define the Hugging Face Hub LLM
hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 500}
)

# Define a query to ask the model
query = """What were the trends in median household income across different states in the United States between 2021 and 2022?"""

# Invoke the model and print the result
response = hf.invoke(query)
print(response)
# Use similarity searching algorithm and return 3 most relevant documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
import huggingface_hub
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" #Token with Read permissions 

# Define the Hugging Face Hub LLM
hf = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-v0.1",
    model_kwargs={"temperature": 0.1, "max_length": 500}
)

# Define a query to ask the model
query = """What were the trends in median household income across different states in the United States between 2021 and 2022?"""

# Invoke the model and print the result
response = hf.invoke(query)
print(response)
