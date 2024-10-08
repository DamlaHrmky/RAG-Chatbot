import streamlit as st
from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from dotenv import load_dotenv
import os
import shutil

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI and Tavily API keys from environment variables
api_key = os.getenv('OPENAI_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')


# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Function to clear the static folder
def clear_static_folder():
    """Remove all files in the static folder."""
    for filename in os.listdir("static"):
        file_path = os.path.join("static", filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Function to load ChromaDB connection and create index
@st.cache_resource
def load_chroma(file_path):
    # Load PDF document from the saved file path
    pdf_doc = read_doc(file_path)
    #bos listeye eklicek
    # Split the PDF content into chunks
    pdf_doc = chunk_data(docs=pdf_doc)
    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072, openai_api_key=api_key)
    # Create Chroma index from documents
    index = Chroma.from_documents(documents=pdf_doc, embedding=embeddings, persist_directory="./vectorstore")
    return index

# PDF reading function for file path
def read_doc(file_path):
    # Use PyPDFium2Loader with the file path
    file_loader = PyPDFium2Loader(file_path)
    pdf_documents = file_loader.load()
    return pdf_documents

# Chunk splitting function
def chunk_data(docs, chunk_size=1000, chunk_overlap=200):
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pdf = text_splitter.split_documents(docs)
    return pdf

# Function to retrieve relevant chunks for the given query
def retrieve_query(query, k=5, index=None):
    # Create a retriever to fetch relevant documents
    retriever = index.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# Function to generate answers based on the query
def get_answers(query, k=5, index=None):
    # Retrieve relevant documents
    doc_search = retrieve_query(query, k=k, index=index)
    # Define a template for the response
    template = """Use the following pieces of context to answer the user's question of "{question}".
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
"{context}" """
    # Set up the prompt template
    prompt_template = PromptTemplate(input_variables=['question', 'context'], template=template)
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, top_p=1)
    # Chain together the prompt and language model
    chain = prompt_template | llm | StrOutputParser()
    # Generate output using the chained prompt and model
    output = chain.invoke({"question": query, "context": doc_search})
    return output

# Function to summarize the PDF document (Placeholder)
def summarize_pdf(index):
    # Define a simple summarization template or prompt here
    summary_prompt = """Summarize the following document into key points."""
    # Implement summarization logic here, depending on your use case
    # For simplicity, let's return a placeholder
    return "Summary functionality is under development."

# Function to inject custom CSS for sidebar
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Change sidebar background color to a darker shade */
        .css-1l02x9p {
            background-color:#1e1e1e ; /* Very dark gray for the sidebar background */
        }
        /* Change sidebar text color to a lighter shade for better contrast */
        .css-1l02x9p .css-1v3fvcr {
            color: #dcdcdc; /* Light gray text color */
        }
        /* Adjust other elements if needed */
        .css-1l02x9p .css-1v3fvcr a {
            color: #87ceeb; /* Light blue color for links */
        }
        .css-1l02x9p .css-1v3fvcr a:hover {
            color: #b0e0e6; /* Lighter blue on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit app layout
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Inject custom CSS
inject_custom_css()
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff000050;
    }
</style>
""", unsafe_allow_html=True)

# Navigation menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ("Chat", "Summarization"))

# Sidebar file uploader component for uploading PDFs
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Initialize agent and tools
search_tool = TavilySearchResults(max_results=1)
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini", top_p=1.0)
prompt = hub.pull("hwchase17/openai-tools-agent", api_key=langchain_api_key)
prompt.messages[0].prompt.template = (
    "Make sure to use the bert_and_language_understanding_tool for questions about the BERT model, "
    "and the tavily_search_results_json tool for questions you don't know about."
)
retriever_new = None
tools = [search_tool]
agent_executor = None

# Initialize or load previous Q&A from session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Handle file upload
if uploaded_file is not None:
    # Clear the static folder before saving a new file
    clear_static_folder()

    # Save the uploaded file to the static folder
    save_path = os.path.join("static", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the index using the saved file path
    index = load_chroma(save_path)

    # Create and configure the retriever tool
    retriever_new = index.as_retriever(search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever=retriever_new,
        name="bert_and_language_understanding_tool",
        description="Search for information about the BERT model and language understanding. For any questions related to the article 'Bert Model', you must use this tool!"
    )
    tools.append(retriever_tool)
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Display based on selected page
    if page == "Chat":
        st.title("Chat Functionality")

        def ask_question():
            with st.form(key='question_form', clear_on_submit=True):
                user_input = st.text_input("Please write your question here:")
                submit_button = st.form_submit_button("Ask")

                # When the user submits a question
                if submit_button and user_input:
                    # First, try to find an answer from the document
                    answer = get_answers(user_input, index=index)
                    # If no answer found in the document, use the agent
                    if not answer or "don't know" in answer.lower():
                        response = agent_executor.invoke({"input": user_input})
                        # Extract and display only the answer text from the agent's response
                        if 'output' in response:
                            answer_text = response['output']
                        else:
                            answer_text = "The agent could not provide an answer."
                    else:
                        answer_text = answer
                    
                    # Add new Q&A to the list
                    st.session_state.qa_history.insert(0, {'question': user_input, 'answer': answer_text})
                    
                    # Limit the history to the last 3 items
                    st.session_state.qa_history = st.session_state.qa_history[:]
                    
                    # Display the last 3 Q&A pairs
                    for qa in st.session_state.qa_history:
                        st.write(f"**Question:** {qa['question']}")
                        st.write(f"**Answer:** {qa['answer']}")

        # Display the question box for user input
        ask_question()

    elif page == "Summarization":
        st.title("Document Summarization")
        summary = summarize_pdf(index)
        st.write(summary)

else:
    st.write("Please upload a PDF file to start.")