# TAVILY_API_KEY="": Login: https://app.tavily.com/home

from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
import os
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tools = [TavilySearch()]
llm = ChatOllama(temperature=0, model="phi3:mini")
tracer_provider = register(
        endpoint="http://localhost:6006/v1/traces"  # phoenix serve endpoint
    )

# instrument langchain
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
chain = agent_executor

def main():

    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()