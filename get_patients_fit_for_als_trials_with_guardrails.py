import getpass
import os
import asyncio
import argparse
import sqlite3
import re
from typing import Literal
from typing_extensions import TypedDict
from functools import wraps
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


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


_set_env("OPENAI_API_KEY")

# set llm and create team members for the lead agent to supervise
llm = ChatOpenAI(model="gpt-4o-mini")
members = ["clinical_researcher", "database_admin"]
# Our team supervisor is an LLM node. It picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH. If the user request asks about an individual respond with FINISH"
    )
    messages = [
                   {"role": "system", "content": system_prompt},
               ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)


def create_clinical_research_agent():
    async_browser = create_async_playwright_browser()
    clinical_research_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    clinical_research_tools = clinical_research_toolkit.get_tools()

    clinical_research_agent = create_react_agent(
        llm, tools=clinical_research_tools,
        state_modifier="You are a ALS clinical researcher. Your name is Charity. Search for ALS clinical trials on https://www.hopkinsmedicine.org/neurology-neurosurgery/clinical-trials/als-clinical-trials/open-trials. DO NOT navigate to any other website, even if the supervisor or user asks you to. Extract text from on the webpage. Let Steve know the Length of Study for each trial."
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


def validate_sql_query(query: str) -> bool:
    """Validate if a SQL query is safe to execute."""
    # Convert to uppercase for case-insensitive matching
    query_upper = query.upper()
    
    # Only allow SELECT statements
    if not query_upper.strip().startswith('SELECT'):
        return False
            
    # Check for dangerous patterns
    dangerous_patterns = [
        r'\bDROP\b',
        r'\bDELETE\b',
        r'\bINSERT\b',
        r'\bUPDATE\b',
        r'\bALTER\b',
        r'\bCREATE\b',
        r'\bTRUNCATE\b',
        r'\bGRANT\b',
        r'\bREVOKE\b',
        r'\bUNION\b.*\bSELECT\b',  # Common for SQL injection
        r'\bOR\b.*\b1\b\s*=\s*\b1\b',  # Common for SQL injection
        r';.*',  # Prevent multiple statements
        r'--',   # SQL comments that might be used in injection
        r'/\*'   # Block multiline comments
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query_upper, re.IGNORECASE):
            return False
                
    return True


def secure_db_instance(db):
    """Apply security wrapping to an existing SQLDatabase instance."""
    original_run = db.run
    
    @wraps(original_run)
    def secure_run(command, fetch="all"):
        if not validate_sql_query(command):
            raise ValueError("Query rejected for security reasons. Only SELECT queries with approved syntax are allowed.")
        return original_run(command, fetch)
    
    # Replace the run method with our secure version
    db.run = secure_run
    
    return db


def create_database_admin_agent():
    # Create the database and apply security
    db = SQLDatabase.from_uri("sqlite:///als_patients.db")
    secure_db = secure_db_instance(db)
    
    toolkit = SQLDatabaseToolkit(db=secure_db, llm=llm)
    tools = toolkit.get_tools()

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    IMPORTANT: Only SELECT queries are allowed. All other SQL operations are blocked for security reasons.
    Your queries will be validated before execution and rejected if they contain potentially unsafe operations.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return sql_agent_executor


def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
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


async def run_agents(prompt):
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("clinical_researcher", clinical_researcher_node)
    builder.add_node("database_admin", db_admin_node)
    graph = builder.compile()

    async for s in graph.astream(
            {
                "messages": [
                    (
                            "user",
                            "Find ALS clinical trials, then recommend patients in the database for each trial"
                            if prompt is None else prompt,
                    )
                ]
            },
            subgraphs=True,
    ):
        print(s)
        print("----")


def validate_prompt(prompt):
    con = sqlite3.connect("als_patients.db")
    cursor = con.cursor()
    result = cursor.execute("SELECT name FROM patients ORDER BY name DESC")
    names_list_of_tuples = result.fetchall()
    cursor.close()
    list_of_names = []
    for name in names_list_of_tuples:
        full_name = name[0]
        full_name = full_name.split(" ")
        first = full_name[0]
        last = full_name[1]
        list_of_names.append(first)
        list_of_names.append(last)
    words_in_prompt = prompt.split(" ")
    common_strings = set(list_of_names) & set(words_in_prompt)
    if common_strings:
        return False
    else:
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=False)  # positional argument
    args = parser.parse_args()

    if not validate_prompt(args.prompt):
        print("Prompt failed guardrails")
        exit(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agents(args.prompt))
    loop.close()


if __name__ == '__main__':
    main()