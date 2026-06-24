import os
from dotenv import load_dotenv

# MUST be before CrewAI imports
os.environ["CREWAI_TELEMETRY"] = "false"
load_dotenv()

import streamlit as st
from crewai import Crew, Process, Task, Agent
from crewai_tools import SerperDevTool, tool
from tools import document_search_tool


def build_llm():
    """Support both newer CrewAI LLM and older LangChain Ollama paths."""
    try:
        from crewai import LLM as CrewLLM
        return CrewLLM(model="ollama/llama3.2", base_url="http://localhost:11434")
    except Exception:
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model="llama3.2", base_url="http://localhost:11434")
        except Exception:
            from langchain_community.llms import Ollama
            return Ollama(model="llama3.2", base_url="http://localhost:11434")

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Indian Criminal Law RAG Agent",
    page_icon=":scales:",
    layout="wide"
)

st.title("Indian Criminal Law RAG Agent")
st.markdown("""
This agent provides legal question answering, section lookup, case reasoning,
and contextual explanations related to Indian Criminal Law (BNS, BNSS, IPC, CrPC).
""")

# ------------------------------
# Sidebar Configuration
# ------------------------------
with st.sidebar:
    st.header("Configuration")
    st.info("Ensure Ollama is running locally with LLaMA 3.2.")
    debug_mode = st.checkbox("Show Agent Reasoning Steps", value=True)
    
    # Optional Serper API key for web search
    serper_api_key = st.text_input("Serper API Key (Optional for web search)", type="password")
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key

# ------------------------------
# Optional Web Search Wrapper
# ------------------------------
serper_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None


@tool("Search the internet")
def search_internet_tool(query: str):
    """
    Web search wrapper that accepts `query` and maps it to Serper input.
    """
    if serper_tool is None:
        return "Web search is disabled because SERPER_API_KEY is not set."

    # Handle different SerperDevTool call signatures across versions.
    for call in (
        lambda: serper_tool.run(search_query=query),
        lambda: serper_tool.run(query=query),
        lambda: serper_tool.run(query),
        lambda: serper_tool._run(search_query=query),  # noqa: SLF001
        lambda: serper_tool._run(query=query),  # noqa: SLF001
        lambda: serper_tool._run(query),  # noqa: SLF001
    ):
        try:
            return call()
        except Exception:
            continue

    return "Web search failed due to incompatible Serper tool signature."

# ------------------------------
# Initialize Ollama LLM
# ------------------------------
llm = build_llm()

# ------------------------------
# Agents
# ------------------------------
retrieval_agent = Agent(
    role="Legal Document Retrieval Specialist",
    goal="Efficiently retrieve relevant legal document chunks from the local vector database based on user queries.",
    backstory="You are a specialist in information retrieval, specifically for legal documents. Your primary responsibility is to interface with the local Qdrant vector database to find the most semantically relevant sections of the IPC, CrPC, BNS, and BNSS.",
    verbose=True,
    tools=[document_search_tool],
    llm=llm,
    allow_delegation=False,
    max_iter=4
)

research_agent = Agent(
    role="Legal Research Specialist",
    goal="Conduct thorough research on Indian Criminal Law (BNS, BNSS, IPC, CrPC) and case laws to find relevant legal provisions and precedents.",
    backstory="You are an expert legal researcher with deep knowledge of Indian criminal statutes. Your expertise lies in navigating complex legal documents and identifying the most relevant sections and case laws for any given legal query.",
    verbose=True,
    tools=[search_internet_tool] if os.getenv("SERPER_API_KEY") else [],
    llm=llm,
    allow_delegation=False,
    max_iter=4
)

legal_reasoning_agent = Agent(
    role="Legal Reasoning and Analysis Expert",
    goal="Analyze the retrieved legal information and provide clear, concise, and contextually relevant explanations and reasoning for legal questions.",
    backstory="You are a seasoned legal analyst specializing in Indian Criminal Law. Your role is to take the raw legal data provided by the Research Agent and synthesize it into a coherent legal argument or explanation.",
    verbose=True,
    llm=llm,
    allow_delegation=False,
    max_iter=4
)

# ------------------------------
# User Input
# ------------------------------
query = st.text_input("Ask a legal question:", placeholder="e.g., What is the punishment for murder in BNS?")
submit = st.button("Get Answer")

if submit and query:
    with st.spinner("Agent is thinking..."):
        try:
            # Define tasks
            retrieval_task = Task(
                description=f"Retrieve the most relevant legal document chunks from the local vector database for the following user query: {query}.",
                expected_output="A list of relevant legal document chunks with their sources and content.",
                agent=retrieval_agent
            )

            research_task = Task(
                description=(
                    f"Conduct focused research for: {query}. "
                    "Use web search only if strictly required. "
                    "When searching, pass a plain text query string."
                ),
                expected_output="A comprehensive report summarizing the relevant legal provisions, case laws, and precedents.",
                agent=research_agent
            )

            legal_reasoning_task = Task(
                description=f"Analyze the research report and provide a clear, concise, contextually relevant explanation and reasoning for the query: {query}.",
                expected_output="A final legal explanation and reasoning for the query, including references to relevant sections and case laws.",
                agent=legal_reasoning_agent
            )

            # Create crew
            crew = Crew(
                agents=[retrieval_agent, research_agent, legal_reasoning_agent],
                tasks=[retrieval_task, research_task, legal_reasoning_task],
                process=Process.sequential,
                verbose=True
            )

            # Run crew
            result = crew.kickoff(inputs={"query": query})
            st.subheader("Answer:")
            st.write(getattr(result, "raw", result))

            if debug_mode:
                st.subheader("Retrieved Sources:")
                st.info("Sources were retrieved from the local Qdrant database.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.info("Ensure Ollama LLaMA 3.2 is running locally and Qdrant DB is accessible.")

elif submit and not query:
    st.warning("Please enter a question.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Built with CrewAI, Qdrant, and Ollama.")

