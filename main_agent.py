from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
#from tavily import TavilyClient

#tavily = TavilyClient()  # Initialize Tavily client
from langchain_tavily import TavilySearch




# @tool
# def search(query: str) -> str:
#     # Placeholder for search functionality
#     """
#     Tool that searches over internet
#     Args:
#         query: The search query
#     Returns:
#         The search result
#     """
#     print(f"Searching for: {query}")
#     return tavily.search(query=query, max_results=1,topic="general")  # Use Tavily to perform the search


llm = ChatOpenAI(model="gpt-5.5", temperature=0)
tools = [TavilySearch()]
#tools = [search]

agent = create_agent(model=llm, tools=tools)


def main_agent() -> None:
    print("Welcome to the LangChain Agent!")
    print("Your OpenAI API Key is:", bool(os.getenv("OPENAI_API_KEY")))
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 Job postings for AI engineer in Singapore")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main_agent()
