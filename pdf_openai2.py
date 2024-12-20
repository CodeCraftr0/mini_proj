import os
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets
from urllib.request import urlretrieve
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Styling
display(HTML("""
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
.message {
    padding: 10px 15px;
    margin: 5px;
    border-radius: 15px;
    max-width: 70%;
    word-wrap: break-word;
}
.user-message {
    background-color: #e3f2fd;
    margin-left: auto;
    text-align: right;
}
.bot-message {
    background-color: #f5f5f5;
    margin-right: auto;
    text-align: left;
}
</style>
"""))

# Function to preprocess PDFs
def setup_rag_system():
    # Download and load PDFs
    os.makedirs("legal_doc", exist_ok=True)
    files = [
        "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a2023-45.pdf",
        "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf",
    ]
    for url in files:
        file_path = os.path.join("legal_doc", url.rpartition("/")[2])
        urlretrieve(url, file_path)

    loader = PyPDFDirectoryLoader("./legal_doc/")
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=OpenAIEmbeddings()
    )

    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Define prompt template
    prompt_template = """
    You are an expert assistant. Use the following context to answer the question accurately and concisely:

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"], template=prompt_template
    )

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# Initialize RAG system
rag_chain = setup_rag_system()

# Chat Interface
class ChatInterface:
    def __init__(self, rag_chain):
        self.messages = []
        self.rag_chain = rag_chain

        # Widgets
        self.text_input = widgets.Text(
            placeholder='Ask a question...', layout=widgets.Layout(width='70%')
        )
        self.send_button = widgets.Button(
            description='Send', button_style='primary', layout=widgets.Layout(width='10%')
        )
        self.clear_button = widgets.Button(
            description='Clear Chat', button_style='danger', layout=widgets.Layout(width='10%')
        )
        self.chat_output = widgets.Output()

        # Layout
        self.input_box = widgets.HBox(
            [self.text_input, self.send_button, self.clear_button]
        )

        # Button Handlers
        self.send_button.on_click(self.on_send)
        self.clear_button.on_click(self.on_clear)
        self.text_input.on_submit(self.on_send)

    def on_send(self, b=None):
        question = self.text_input.value.strip()
        if question:
            self.add_message(question, "user")
            with self.chat_output:
                # Get answer from RAG chain
                response = self.rag_chain.invoke(question)
                self.add_message(response, "bot")
            self.text_input.value = ""

    def on_clear(self, b):
        with self.chat_output:
            clear_output()
        self.messages = []

    def add_message(self, message, sender):
        sanitized_message = message.replace("<", "&lt;").replace(">", "&gt;")
        self.messages.append({"text": sanitized_message, "sender": sender})

        with self.chat_output:
            clear_output()
            for msg in self.messages:
                msg_class = "user-message" if msg["sender"] == "user" else "bot-message"
                display(HTML(f"""
                    <div class="chat-container">
                        <div class="message {msg_class}">
                            {msg['text']}
                        </div>
                    </div>
                """))

    def display(self):
        display(widgets.VBox([self.chat_output, self.input_box]))

# Create and display the chat interface
chat = ChatInterface(rag_chain)
chat.display()
