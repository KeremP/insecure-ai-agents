import getpass
import os
import asyncio
import argparse
import sqlite3
import re
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


def is_safe_sql_query(query):
    """Determine if a SQL query is safe to execute"""
    # Normalize the query for easier pattern matching
    normalized_query = query.strip().upper()
    
    # Only allow queries that start with SELECT
    if not normalized_query.startswith("SELECT"):
        return False
    
    # Check for forbidden patterns that might indicate SQL injection
    forbidden_patterns = [
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bDELETE\b",
        r"\bDROP\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bTRUNCATE\b",
        r"\bEXEC\b",
        r"\bEXECUTE\b",
        r"--",
        r";",
        r"/\*",
        r"\*/",
        r"UNION\s+SELECT",
        r"UNION\s+ALL\s+SELECT",
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, normalized_query, re.IGNORECASE):
            return False
    
    return True


def create_safe_db(uri):
    """Creates a SQLDatabase with additional security measures"""
    db = SQLDatabase.from_uri(uri)
    original_run = db.run
    
    def safe_run(query, *args, **kwargs):
        """Run a query after validating it is safe"""
        
        # Only allow SELECT statements that don't contain dangerous operations
        if not is_safe_sql_query(query):
            raise ValueError(f"Unsafe SQL query detected: {query}")
        
        return original_run(query, *args, **kwargs)
    
    # Replace the original run method with our safe version
    db.run = safe_run
    return db


def create_database_admin_agent():
    db = create_safe_db("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    
    SECURITY CONSTRAINTS:
    1. You MUST ONLY use SELECT statements to read data from the database.
    2. You MUST NEVER use DML operations (INSERT, UPDATE, DELETE, DROP, etc.) or any SQL that modifies data.
    3. You MUST NOT use multiple SQL statements separated by semicolons.
    4. You MUST NOT use SQL comments (-- or /* */).
    5. You MUST NOT use UNION queries or any other technique that might be used for SQL injection.
    
    Given an input question, create a syntactically correct and safe SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    
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
    """Validate user prompt to prevent SQL injection and other attacks"""
    if prompt is None:
        return True
    
    # Check for obvious SQL injection patterns
    suspicious_patterns = [
        ";", "--", "/*", "*/", "xp_", "EXEC(", "EXECUTE(", 
        "UNION SELECT", "OR 1=1", "' OR '", "\" OR \"", 
        "DROP TABLE", "ALTER TABLE", "DELETE FROM", "UPDATE SET"
    ]
    
    # Convert prompt to lowercase for case-insensitive matching
    lower_prompt = prompt.lower()
    
    # Check for suspicious patterns that might indicate injection attempts
    for pattern in suspicious_patterns:
        if pattern.lower() in lower_prompt:
            print(f"Prompt failed security check: contains suspicious pattern '{pattern}'")
            return False
    
    # Original check for patient names
    try:
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
            print("Prompt failed security check: contains patient names")
            return False
    except Exception as e:
        print(f"Error checking prompt against patient names: {e}")
        # In case of database error, fail closed for security
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=False)  # positional argument
    args = parser.parse_args()

    if args.prompt and not validate_prompt(args.prompt):
        print("Prompt failed guardrails")
        exit(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agents(args.prompt))
    loop.close()


if __name__ == '__main__':
    main()