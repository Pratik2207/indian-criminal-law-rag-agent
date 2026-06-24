import os
from pathlib import Path
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from tools import document_search_tool

# Initialize the tools
doc_search_tool = document_search_tool
web_search_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None

# Define the LLM (Ollama LLaMA 3.2)
# Note: In a real local setup, this would point to a local Ollama instance.
# For this implementation, we'll configure it to use the local Ollama endpoint.

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


local_llm = build_llm()


@CrewBase
class IndianLawRagCrew():
    """IndianLawRag crew"""

    agents_config = str(Path(__file__).parent / "config" / "agents.yaml")
    tasks_config = str(Path(__file__).parent / "config" / "tasks.yaml")

    @agent
    def retrieval_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['retrieval_agent'],
            verbose=True,
            tools=[doc_search_tool],
            llm=local_llm
        )

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['research_agent'],
            verbose=True,
            tools=[web_search_tool] if web_search_tool else [],
            llm=local_llm
        )

    @agent
    def legal_reasoning_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['legal_reasoning_agent'],
            verbose=True,
            llm=local_llm
        )

    @task
    def retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieval_task'],
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def legal_reasoning_task(self) -> Task:
        return Task(
            config=self.tasks_config['legal_reasoning_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the IndianLawRag crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

if __name__ == "__main__":
    # Test the crew (this will fail if Ollama is not running locally)
    # But we'll provide it as part of the project structure.
    pass
