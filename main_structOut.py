from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent."""

    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response and sources"""

    answer:str = Field(description="The answer provided by the agent")
    sources:List[Source] = Field(description="List of sources used by the agent to generate the answer")

llm = ChatOpenAI(model="gpt-5.5", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main_structOut() -> None:
    print("Welcome to the LangChain Agent!")
    print("Your OpenAI API Key is:", bool(os.getenv("OPENAI_API_KEY")))
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for 3 Job postings for AI engineer in Singapore")]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main_structOut()