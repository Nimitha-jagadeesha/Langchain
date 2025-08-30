from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Phoenix + OpenInference imports
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

load_dotenv()

def main():
    # register tracer provider with phoenix server
    tracer_provider = register(
        endpoint="http://localhost:6006/v1/traces"  # phoenix serve endpoint
    )

    # instrument langchain
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    information = """
    Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman, ...
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    # now no need to pass callbacks, it's auto-instrumented
    response = chain.invoke({"information": information})

    print(response.content)

if __name__ == "__main__":
    main()




