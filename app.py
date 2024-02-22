import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
from utils import *  # Assuming 'utils' module is defined

file_path = 'pdf information.pdf'
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata
def load_docs(file_path):
    try:
        # Extract text from the PDF file
        with fitz.open(file_path) as pdf_document:
            documents = [Document(page.get_text(), metadata={'page_number': i + 1}) for i, page in enumerate(pdf_document)]
        return documents
    except FileNotFoundError:
        print(f"File not found: '{file_path}'")
        return []
documents = load_docs(file_path)
print(len(documents))
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    # Use the RecursiveCharacterTextSplitter with the Document objects
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs
docs = split_docs(documents)

# Initialize session state keys
if 'responses' not in st.session_state:
    st.session_state['responses'] = []
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
# Rest of your code continues here...
st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know'"""
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
st.title("Langchain Chatbot")
response_container = st.container()
textcontainer = st.container()
with textcontainer:
    query = st.text_input("Query: ", key="input")
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
if query:
    with st.spinner("typing..."):
        context = "your_context_here"
        response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
    st.session_state.requests.append(query)
    st.session_state.responses.append(response)












