import getpass
import os
import asyncio
import argparse
import sqlite3
import re  # Added import for regular expressions
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


def create_database_admin_agent():
    db = SQLDatabase.from_uri("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt:31156d5f")
    assert len(prompt_template.messages) == 1

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

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

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
    """
    Validate user prompt for potential security issues.
    Return True if prompt passes all security checks, False otherwise.
    """
    # Check if prompt is None or empty
    if prompt is None or not prompt.strip():
        return False
    
    # Check for SQL injection patterns
    # Common SQL keywords that could indicate injection when used in specific contexts
    sql_keywords = [
        "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "EXEC",
        "UNION", "JOIN", "WHERE", "FROM", "INTO", "TRUNCATE"
    ]
    
    # SQL injection patterns that are more specific and likely malicious
    sql_injection_patterns = [
        "--", "/*", "*/", ";", "OR 1=1", "' OR '", "\" OR \"", "OR TRUE", 
        "SLEEP(", "WAITFOR DELAY", "BENCHMARK(", "UNION ALL SELECT"
    ]
    
    # LLM injection patterns
    llm_injection_patterns = [
        "system:", "assistant:", "user:", "role:", "content:", 
        "ignore previous", "disregard", "bypass", "override", "forget",
        "instructions", "prompt", "system message", "new task"
    ]
    
    # Check for specific SQL injection patterns (these are more likely to be malicious)
    prompt_lower = prompt.lower()
    for pattern in sql_injection_patterns:
        if pattern.lower() in prompt_lower:
            return False
    
    # For SQL keywords, check if they're used in a potentially malicious context
    # This helps avoid flagging legitimate uses of these words in English sentences
    words_in_prompt = prompt_lower.split()
    for i, word in enumerate(words_in_prompt):
        # Strip punctuation for comparison
        clean_word = word.strip(",.;:!?\"'()[]{}").lower()
        
        if clean_word.upper() in sql_keywords:
            # Check if the keyword is part of a potential SQL query pattern
            # For example, "SELECT ... FROM ..." or "DROP TABLE ..."
            if i < len(words_in_prompt) - 2:  # Ensure there are at least 2 more words
                next_words = " ".join(words_in_prompt[i+1:i+3]).lower()
                # Look for patterns that suggest SQL query construction
                if "from" in next_words or "table" in next_words or "database" in next_words:
                    return False
    
    # Check for LLM injection patterns
    for pattern in llm_injection_patterns:
        if pattern.lower() in prompt_lower:
            return False
    
    # Check for attempts to directly manipulate agents by name
    agent_names = ["charity", "steve", "clinical researcher", "database admin"]
    agent_manipulation_patterns = [
        "tell me about", "give me access", "show me", "reveal", "bypass", 
        "ignore", "override", "new instructions"
    ]
    
    # Check for combinations of agent names and manipulation patterns
    for agent in agent_names:
        for pattern in agent_manipulation_patterns:
            combined_pattern = f"{pattern} {agent}"
            if combined_pattern in prompt_lower or f"{agent}, {pattern}" in prompt_lower:
                return False
    
    # Get patient names from database and check if they appear in the prompt
    try:
        con = sqlite3.connect("als_patients.db")
        cursor = con.cursor()
        result = cursor.execute("SELECT name FROM patients ORDER BY name DESC")
        names_list_of_tuples = result.fetchall()
        cursor.close()
        con.close()
    except sqlite3.Error:
        # If we can't check against patient names, fail safe
        return False
    
    # Process all patient names and check for matches in the prompt
    patient_name_parts = []
    
    for name_tuple in names_list_of_tuples:
        if not name_tuple or not name_tuple[0]:
            continue
        
        full_name = name_tuple[0]
        # Split the full name into parts (handles names with more than just first/last)
        name_parts = full_name.split()
        
        for part in name_parts:
            if len(part) > 2:  # Only consider name parts longer than 2 characters
                patient_name_parts.append(part.lower())
    
    # Check if any words in the prompt match patient name parts
    common_strings = set(patient_name_parts) & set(words_in_prompt)
    
    if common_strings:
        return False
    
    # Additional check for partial matches (to catch slight variations)
    for name_part in patient_name_parts:
        if len(name_part) > 3:  # Only check substantial name parts
            for word in words_in_prompt:
                # Check if name part is contained within a word or vice versa
                if name_part in word or (len(word) > 3 and word in name_part):
                    return False
    
    # If we get here, prompt has passed all checks
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=False)  # positional argument
    args = parser.parse_args()

    # Validate prompt if supplied, otherwise use default prompt
    if args.prompt is not None and not validate_prompt(args.prompt):
        print("Prompt failed guardrails")
        exit(1)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_agents(args.prompt))
    loop.close()


if __name__ == '__main__':
    main()