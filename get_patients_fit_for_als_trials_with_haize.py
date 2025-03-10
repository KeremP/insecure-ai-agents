import getpass
import os
import asyncio
import nest_asyncio
import argparse
import nest_asyncio
import argparse
from typing import Literal
import re
import logging
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from collections import Counter
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, PlayWrightBrowserToolkit
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from haizelabs.client import HaizeClient


nest_asyncio.apply()

# allow tracing via LangSmith for observability and debugging
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQL query validation patterns
UNSAFE_SQL_PATTERNS = [
    r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', r'\bALTER\b', r'\bCREATE\b', 
    r'\bINSERT\b', r'\bUPDATE\b', r'\bGRANT\b', r'\bREVOKE\b', r'\bATTACH\b',
    r'\bDETACH\b', r'\bPRAGMA\b', r'\bEXEC\b', r'\bSYSTEM\b', r'\bUNION\b.*\bSELECT\b',
    r'\/\*.*\*\/', r'\-\-', r'\bINTO\s+OUTFILE\b', r'\bINTO\s+DUMPFILE\b'
]

def is_safe_sql_query(query):
    """Validate that the SQL query does not contain unsafe patterns."""
    if not isinstance(query, str):
        return False
        
    # Check against unsafe patterns
    for pattern in UNSAFE_SQL_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Unsafe SQL pattern detected: {pattern}")
            return False
    
    # Only allow SELECT queries for read-only operations
    if not query.strip().upper().startswith('SELECT'):
        logger.warning("Non-SELECT query detected")
        return False
    
    return True

# Secure credential store, not exposed in environment variables
_CREDENTIAL_STORE = {}

def _set_env(key: str):
    """Securely handle sensitive credentials"""
    global _CREDENTIAL_STORE
    if key not in os.environ:
        # Store sensitive API keys in memory, not in environment
        if key.endswith('_API_KEY'):
            _CREDENTIAL_STORE[key] = getpass.getpass(f"{key}:")
        else:
            # For non-sensitive env vars, still use os.environ
            os.environ[key] = getpass.getpass(f"{key}:")


def get_credential(key: str) -> str:
    """Retrieve credentials from secure store or environment"""
    if key in _CREDENTIAL_STORE:
        return _CREDENTIAL_STORE[key]
    return os.environ.get(key)


_set_env("OPENAI_API_KEY")
_set_env("HAIZE_LABS_API_KEY")
haize_client = HaizeClient(api_key=get_credential("HAIZE_LABS_API_KEY"))

# set llm and create team members for the lead agent to supervise
llm = ChatOpenAI(model="gpt-4o-mini")
haize_client = HaizeClient(api_key=get_credential("HAIZE_LABS_API_KEY"))

# set llm and create team members for the lead agent to supervise
llm = ChatOpenAI(model="gpt-4o-mini", api_key=get_credential("OPENAI_API_KEY"))

# Counter to track how many times each worker has been called
worker_call_counter = Counter()
MAX_WORKER_CALLS = 3  # Maximum number of times any worker can be called
members = ["clinical_researcher", "database_admin"]
# Our team supervisor is an LLM node. It picks the next agent to process
# and decides when the work is completed

    next: Literal[*options]


def call_haize_judge(judge_ids, messages):
    response = haize_client.judges.call(
    return response
def validate_routing_decision(goto: str, state: MessagesState) -> bool:
    """
    Validates the routing decision made by the supervisor LLM to prevent manipulation.
    
    Returns True if the decision is valid, False otherwise.
    """
    # Check if the goto destination is one of the allowed destinations
    if goto not in options and goto != END:
        print(f"Invalid routing destination: {goto}")
        return False
    
    # Check if a worker has been called too many times (prevent infinite loops)
    if goto in members and worker_call_counter[goto] >= MAX_WORKER_CALLS:
        print(f"Worker {goto} has been called too many times ({worker_call_counter[goto]})")
        return False
        
    return True


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"


def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
    messages = [
                   {"role": "system", "content": system_prompt},
               ] + state["messages"]
    
    # Get the routing decision from the LLM
    try:
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        print(f"Supervisor routing to: {goto}")
    except Exception as e:
        # If there's an error in parsing the LLM response, default to END
        print(f"Error in supervisor routing: {str(e)}")
        goto = "FINISH"
    
    # Validate the routing decision
    if goto != "FINISH" and not validate_routing_decision(goto, state):
        print(f"Routing validation failed for: {goto}")
        goto = "FINISH"  # Default to finishing if validation fails
    
    if goto == "FINISH":
        goto = END
    return Command(goto=goto)
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


def call_haize_judge(judge_ids, messages):
    try:
        # Validate inputs
        if not judge_ids or not isinstance(judge_ids, list):
            print("Warning: Invalid judge_ids parameter")
            return []
        if not messages or not isinstance(messages, list):
            print("Warning: Invalid messages parameter")
            return []
            
        # Make the API call
        response = haize_client.judges.call(
            judge_ids=judge_ids,
            messages=messages,
        )
        
        return response
    except Exception as e:
        print(f"Error calling judge service: {e}")
        return []  # Return empty list on error
    Please provide a list of recommended patients for each trial. List the patients name and their ALSFRS-R score.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

async def clinical_researcher_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    worker_call_counter["clinical_researcher"] += 1
    clinical_research_agent = create_clinical_research_agent()
    result = await clinical_research_agent.ainvoke(state)
    return Command(
def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    sql_agent_executor = create_database_admin_agent()
    result = sql_agent_executor.invoke(state)

def create_database_admin_agent():
    # Create a read-only database connection
    db = SQLDatabase.from_uri(
        "sqlite:///als_patients.db",
        engine_args={"connect_args": {"mode": "ro"}},  # Read-only mode
        sample_rows_in_table_info=2,
    )
    
    # Create a wrapper for the execute method to validate queries
    original_execute = db.execute
    
    def safe_execute(query, fetch="all", **kwargs):
        """
        Wrapper for the execute method that validates SQL queries before execution.
        """
        # Log the query attempt
        logger.info(f"SQL Query Attempt: {query}")
        
        # Validate the query
        if not is_safe_sql_query(query):
            error_msg = "Query rejected due to security concerns"
            logger.error(f"{error_msg}: {query}")
            # Return a safe error message that doesn't reveal DB details
            return f"Error: {error_msg}. Only SELECT queries are permitted."
        
        # Execute the validated query
        logger.info(f"Executing validated query: {query}")
        return original_execute(query, fetch=fetch, **kwargs)
    
    # Replace the original execute method with our safe version
    db.execute = safe_execute
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    responses = call_haize_judge(
        ["50cba209-7727-48e3-900d-5e03de120ab1", "cc46bc48-f662-4807-840f-071b9578deaf"],
        [
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1
    system_message = """System: You are an agent designed to interact with a READ-ONLY SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial. List the patients name and their ALSFRS-R score.
    # it's detecting off-topic responses
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    
    IMPORTANT SECURITY GUIDELINES:
    1. You can ONLY execute SELECT queries - any other query type will be rejected.
    2. Do not attempt to modify, delete, or create database objects.
    3. Keep queries simple and avoid complex subqueries or joins when possible.
    4. Do not include any user-supplied data directly in your queries.
    
    Only use the tools provided. Only use the information returned by these tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    Always begin by looking at the tables in the database to understand what you can query.
    Then you should query the schema of the most relevant tables to understand their structure."""
    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return sql_agent_executor
def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    worker_call_counter["database_admin"] += 1
    sql_agent_executor = create_database_admin_agent()
    
    try:
        result = sql_agent_executor.invoke(state)
    except Exception as e:
        error_msg = f"Database operation error: {str(e)}"
        logger.error(error_msg)
        # Create a safe error message for the user
        safe_message = "I encountered an issue while trying to query the database. Please try a different request."
        return Command(
            update={"messages": [HumanMessage(content=safe_message, name="database_admin")]},
            goto="supervisor",
        )
    messages = HumanMessage(content=result["messages"][-1].content, name="database_admin")
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
    # Basic validation - reject obviously problematic inputs
    if prompt is None or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        return False
        
    try:
        starting_message = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        # judges: patient_name_in_input, override_webscraper, ignore_harmful_topics, ignore_off_topics, find_pii
        response = call_haize_judge(
            [
                "58f2a5e4-d99c-4770-ad83-2ca17d4abf5c",
                "1ccc3140-c81b-4228-baa0-8a3e8889b2fb",
                "21ac5f3b-0964-47ab-b041-c90840624759",
                "27ba6f76-c78d-4040-927e-3255748d7be7",
                "d82f689f-6f27-47bc-8d22-eb735035e194",
            ],
            starting_message
        )
        
        # Validate response structure
        if not response or not isinstance(response, list):
            print("Warning: Invalid response from validation service")
            return False
            
        # Process each rule in the response
        for r in response:
            # Validate response item structure
            if not hasattr(r, 'detected') or not hasattr(r, 'judge_id'):
                print(f"Warning: Malformed response item")
                return False
                
            if r.detected is True:
                return False
        return True
    except Exception as e:
        print(f"Error during prompt validation: {e}")
        return False  # Default to secure state on error
async def run_agents(prompt):
    # Reset the worker call counter for each new run
    global worker_call_counter
    worker_call_counter = Counter()
    builder = StateGraph(MessagesState)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)