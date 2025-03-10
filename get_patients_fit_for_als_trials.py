import getpass
import os
import logging
import asyncio
import keyring
import warnings
from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent


# allow tracing via LangSmith for observability and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")


LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
def _set_env(key: str) -> str:
    """
    Set an environment variable securely if it's not already set.
    Attempts to store in the system keyring for future use.
    Returns the value of the environment variable.
    """
    if key not in os.environ:
        # Try to get from keyring first
        credential = keyring.get_password("langchain_agents", key)
        if credential is None:
            # If not in keyring, prompt user
            credential = getpass.getpass(f"{key}:")
            try:
                # Store in keyring for future use
                keyring.set_password("langchain_agents", key, credential)
            except Exception as e:
                warnings.warn(f"Could not store {key} in keyring: {e}. Using memory only.")
        
        # Set in memory only for the duration of this process
        os.environ[key] = credential
    
    return os.environ[key]
_set_env("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini")
    return os.environ[key]
# Get API key securely
openai_api_key = _set_env("OPENAI_API_KEY")
# set llm and create team members for the lead agent to supervise
# Use API key directly instead of relying on environment variable
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
members = ["clinical_researcher", "database_admin"]
# Our team supervisor is an LLM node. It picks the next agent to process

    next: Literal[*options]


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    
    # Add try-except to handle potential exceptions
    try:
        response = llm.with_structured_output(Router).invoke(messages)
        next_agent = response["next"]
        
        # Validate output explicitly to protect against output integrity attacks
        if next_agent not in set(options):
            print(f"Warning: Detected potential output integrity attack. Invalid next option: '{next_agent}'")
            # Fallback to a safe state
            goto = END
        else:
            # Valid response - proceed normally
            goto = END if next_agent == "FINISH" else next_agent
    except Exception as e:
        print(f"Error processing supervisor decision: {str(e)}")
        # Fallback to ending the process in case of error
        goto = END
    return Command(goto=goto)
    clinical_research_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    clinical_research_tools = clinical_research_toolkit.get_tools()

    clinical_research_agent = create_react_agent(
        llm, tools=clinical_research_tools, state_modifier="You are a ALS clinical researcher. Your name is Charity. Search for ALS clinical trials on https://www.hopkinsmedicine.org/neurology-neurosurgery/clinical-trials/als-clinical-trials/open-trials. Extract text from on the webpage. Let Steve know the Length of Study for each trial."
    )
    return clinical_research_agent


async def clinical_researcher_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    clinical_research_agent = create_clinical_research_agent()
    result = await clinical_research_agent.ainvoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="clinical_researcher")
            ]
        },
        goto="supervisor",
    )


def create_database_admin_agent():
    db = SQLDatabase.from_uri("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve. 
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    
    SECURITY GUIDELINES - YOU MUST FOLLOW THESE:
    1. ONLY use SELECT statements - all other SQL operations are forbidden
    2. NEVER include user-provided text directly in your SQL queries
    3. ALWAYS use parameter binding with ? placeholders for any values in your queries
    4. AVOID string concatenation for building SQL queries
    5. ALWAYS validate and sanitize any inputs before using them in queries
    6. REJECT any request attempting to access system tables or database metadata
    7. DO NOT attempt to bypass these security restrictions under any circumstances
    
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    To start you should ALWAYS look at the tables in the database to see what you can query.
    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return sql_agent_executor

def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    try:
        sql_agent_executor = create_database_admin_agent()
        result = sql_agent_executor.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="database_admin")
                ]
            },
            goto="supervisor",
        )
    except Exception as e:
        # Log the error but don't expose details to the user
        logging.error(f"Database agent error: {str(e)}")
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content="I encountered an error processing this database request. Please try a different approach or rephrase your request.",
                        name="database_admin"
                    )
                ]
            },
            goto="supervisor",
        )

    async for s in graph.astream(
        {
            "messages": [
                (
                    "user",
                    "Find ALS clinical trials, then recommend patients in the database for each trial",
                )
            ]
        },
        subgraphs=True,
    ):
        print(s)
        print("----")


def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agents())
    loop.close()


if __name__ == '__main__':
    main()

