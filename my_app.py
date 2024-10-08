import streamlit as st
from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI and Tavily API keys from environment variables
api_key = os.getenv('OPENAI_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
langchain_api_key = os.getenv('LANGCHAIN_API_KEY')

# Set the permanent PDF file path
pdf_file_path = "N19-1423.pdf"

# Define the vectorstore directory
vectorstore_directory = "./vectorstore"

# Function to load or create ChromaDB connection and create index
@st.cache_resource
def load_or_create_chroma(file_path, vectorstore_dir):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072, openai_api_key=api_key)
    
    if os.path.exists(vectorstore_dir):
        index = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
        st.write("Loaded existing Chroma index from the directory.")
    else:
        pdf_doc = read_doc(file_path)
        pdf_doc = chunk_data(docs=pdf_doc)
        index = Chroma.from_documents(documents=pdf_doc, embedding=embeddings, persist_directory=vectorstore_dir)
        st.write("Created a new Chroma index and saved it to the directory.")
    
    return index

def read_doc(file_path):
    file_loader = PyPDFium2Loader(file_path)
    pdf_documents = file_loader.load()
    return pdf_documents

def chunk_data(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pdf = text_splitter.split_documents(docs)
    return pdf

def retrieve_query(query, k=5, index=None):
    retriever = index.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# Modified summarization function with summarization type selection
def summarize_pdf_content(pdf_path, summarize_type):
    pdf = read_doc(pdf_path)
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4o-mini',
        max_tokens=600
    )

    # Load the summarization chain according to the selected type
    chain = load_summarize_chain(llm, chain_type=summarize_type)
    
    output_summary = chain.invoke(pdf)['output_text']
    return output_summary

def get_answers(query, k=5, index=None):
    doc_search = retrieve_query(query, k=k, index=index)
    template = """Use the following pieces of context to answer the user's question of "{question}".
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
"{context}" """
    prompt_template = PromptTemplate(input_variables=['question', 'context'], template=template)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, top_p=1)
    chain = prompt_template | llm | StrOutputParser()
    output = chain.invoke({"question": query, "context": doc_search})
    return output

def inject_custom_css():
    st.markdown(
        """
        <style>
        .css-1l02x9p {
            background-color:#1e1e1e;
        }
        .css-1l02x9p .css-1v3fvcr {
            color: #dcdcdc;
        }
        .css-1l02x9p .css-1v3fvcr a {
            color: #87ceeb;
        }
        .css-1l02x9p .css-1v3fvcr a:hover {
            color: #b0e0e6;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="RAG Chatbot", layout="wide")

inject_custom_css()
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff000050;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ("Chat", "Summarization"))

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

if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

index = load_or_create_chroma(pdf_file_path, vectorstore_directory)

retriever_new = index.as_retriever(search_kwargs={"k": 5})
retriever_tool = create_retriever_tool(
    retriever=retriever_new,
    name="bert_and_language_understanding_tool",
    description="Search for information about the BERT model and language understanding."
)
tools.append(retriever_tool)
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if page == "Chat":
    st.title("Chat Functionality")

    def ask_question():
        with st.form(key='question_form', clear_on_submit=True):
            user_input = st.text_input("Please write your question here:")
            submit_button = st.form_submit_button("Ask")

            if submit_button and user_input:
                answer = get_answers(user_input, index=index)
                if not answer or "don't know" in answer.lower():
                    response = agent_executor.invoke({"input": user_input})
                    if 'output' in response:
                        answer_text = response['output']
                    else:
                        answer_text = "The agent could not provide an answer."
                else:
                    answer_text = answer
                
                st.session_state.qa_history.insert(0, {'question': user_input, 'answer': answer_text})
                st.session_state.qa_history = st.session_state.qa_history[:3]
                
                for qa in st.session_state.qa_history:
                    st.write(f"**Question:** {qa['question']}")
                    st.write(f"**Answer:** {qa['answer']}")

    ask_question()

elif page == "Summarization":
    st.title("Document Summarization")

    # Add a radio button for selecting summarization type
    summarize_type = st.radio(
        "Select summarization type:",
        ('stuff', 'map_reduce', 'refine')
    )

    summary_button = st.button("Generate Summary")
    if summary_button:
        with st.spinner("Summarizing..."):
            summary = summarize_pdf_content(pdf_file_path, summarize_type)
            st.success("Summarization complete!")
            st.subheader("Summary")
            st.write(summary)