import getpass
import os
import asyncio
import logging
import nest_asyncio
import argparse
from typing import Literal
import argparse
from typing import Literal
from typing_extensions import TypedDict
import re
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from haizelabs.client import HaizeClient

nest_asyncio.apply()

# Set up logging for SQL query auditing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sql_queries.log"),
        logging.StreamHandler()
    ])
# allow tracing via LangSmith for observability and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")


# Create a secure credentials store
class SecureCredentialsStore:
    """A secure in-memory store for sensitive credentials."""
    
    def __init__(self):
        self._credentials = {}
    
    def set(self, key, value):
        """Store a credential."""
        self._credentials[key] = value
    
    def get(self, key, default=None):
        """Retrieve a credential."""
        return self._credentials.get(key, default)
    
    def __contains__(self, key):
        """Check if a credential exists."""
        return key in self._credentials

# Initialize the secure store
_secure_credentials = SecureCredentialsStore()

def _set_env(key: str):
    if key not in _secure_credentials and key not in os.environ:
        _secure_credentials.set(key, getpass.getpass(f"{key}:"))
    # If it's in the environment already, we'll use that value


_set_env("OPENAI_API_KEY")
_set_env("HAIZE_LABS_API_KEY")
haize_client = HaizeClient(api_key=_secure_credentials.get("HAIZE_LABS_API_KEY") or os.environ.get("HAIZE_LABS_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini", api_key=_secure_credentials.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"))
# Our team supervisor is an LLM node. It picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


def call_haize_judge(judge_ids, messages):
    response = haize_client.judges.call(
        judge_ids=judge_ids,
        messages=messages,
    )
    return response


options = members + ["FINISH"]


# Add a judge for supervisor routing validation
SUPERVISOR_ROUTING_JUDGE_ID = "cc46bc48-f662-4807-840f-071b9578deaf"  # Using an existing judge for simplicity


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

        " respond with FINISH."
    )

def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
        "You are a supervisor agent tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond ONLY with the name of the worker to act next from this exact list:"
        f" {members}. Each worker will perform a task and respond with their results"
        " and status. When you believe the task is complete, respond with FINISH."
        " Do not output anything besides these exact worker names or FINISH."
    )
    
    try:
        messages = [
                   {"role": "system", "content": system_prompt},
               ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages)
        goto = response["next"]
        
        # Validate the LLM response is one of the expected options
        if goto not in options:
            print(f"Warning: LLM returned invalid routing value: {goto}. Defaulting to END.")
            return Command(goto=END)
            
        # Optionally validate with Haize judge
        judge_response = call_haize_judge([SUPERVISOR_ROUTING_JUDGE_ID], [{"role": "assistant", "content": f"Routing to: {goto}"}])
        if any(r.detected and r.judge_id == SUPERVISOR_ROUTING_JUDGE_ID for r in judge_response):
            print(f"Security check failed for routing to {goto}. Defaulting to END.")
            return Command(goto=END)
            
        if goto == "FINISH":
            goto = END
        return Command(goto=goto)
    except Exception as e:
        print(f"Error in supervisor routing: {str(e)}")
        # Safe fallback in case of any exception
        return Command(goto=END)
        goto="supervisor",
    )

def is_safe_sql_query(query):
    """
    Validate a SQL query for safety.
    
    Args:
        query (str): The SQL query to validate
        
    Returns:
        bool: True if the query is safe, False otherwise
    """
    # Convert to lowercase for easier matching
    query_lower = query.lower().strip()
    
    # Only allow SELECT statements (read-only)
    if not query_lower.startswith("select "):
        logging.warning(f"Blocked non-SELECT query: {query}")
        return False
    
    # List of potentially dangerous operations
    dangerous_operations = [
        "drop", "truncate", "delete", "update", "alter",
        "insert", "replace", "create", "grant", "revoke",
        "attach", "detach", "pragma", "--", "/*", "*/"
    ]
    
    # Check for dangerous operations
    for op in dangerous_operations:
        if op in query_lower:
            logging.warning(f"Blocked dangerous query containing '{op}': {query}")
            return False
    
    # Log approved queries for audit
    logging.info(f"Approved query: {query}")
    return True


def create_database_admin_agent():
    db = SQLDatabase.from_uri("sqlite:///als_patients.db")
    
    # Create a custom version of the database that validates queries
    original_run_query = db.run_query
    
    def safe_run_query(query, *args, **kwargs):
        """Wrap the run_query method with validation."""
        logging.info(f"Received query: {query}")
        if not is_safe_sql_query(query):
            return "Error: This query has been blocked for security reasons. Only SELECT statements are allowed."
        
        try:
            return original_run_query(query, *args, **kwargs)
        except Exception as e:
            # Log the actual error but return a generic message
            logging.error(f"Error executing query: {query}\nError: {str(e)}")
            return f"Error executing query: A database error occurred. Please try a different query."
    
    # Replace the run_query method with our safe version
    db.run_query = safe_run_query
    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    
    IMPORTANT SECURITY GUIDELINES:
    - Generate only READ-ONLY queries (SELECT statements)
    - Never attempt to modify the database in any way
    - Do not use multiple statements in a single query
    - Avoid using user input directly in your queries
    Please provide a list of recommended patients for each trial. List the patients name and their ALSFRS-R score.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    assert len(prompt_template.messages) == 1

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
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

    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return sql_agent_executor


def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    sql_agent_executor = create_database_admin_agent()
    result = sql_agent_executor.invoke(state)

    messages = HumanMessage(content=result["messages"][-1].content, name="database_admin")

    # judges: als_trials_more_than_one_patient_in_output_judge -> we want this to actually return True
    # ignore_off_topics_in_chat_output (finds content not related to ALS patients and which trials they are a good
    # fit for -> we want this to return False
    responses = call_haize_judge(
        ["50cba209-7727-48e3-900d-5e03de120ab1", "cc46bc48-f662-4807-840f-071b9578deaf"],
        [
            {
                "role": "assistant",
                "content": messages.content,
            }
        ],
    )
    # it's detecting there's at least two patients in the output so were not giving away individual patient info
    # it's detecting off-topic responses
    for r in responses:
        if r.judge_id == "50cba209-7727-48e3-900d-5e03de120ab1" and not r.detected:
            messages = HumanMessage(content="Please ask a different question. FINISH.", name="supervisor")
        if r.judge_id == "cc46bc48-f662-4807-840f-071b9578deaf" and r.detected:
            messages = HumanMessage(content="Please ask a different question. FINISH.", name="supervisor")
    return Command(
        update={
            "messages": [
                messages
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


        print("----")


def validate_prompt(prompt: str) -> bool:
    """
    Validates user input prompt for safety before sending to judge APIs
    
    Args:
        prompt: The user provided prompt string
        
    Returns:
        bool: True if prompt passes all validation checks, False otherwise
    """
    # Check if prompt is None or empty
    if not prompt or not isinstance(prompt, str):
        return False
        
    # Check prompt length (prevent excessively long prompts)
    MAX_PROMPT_LENGTH = 500
    if len(prompt) > MAX_PROMPT_LENGTH:
        print(f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")
        return False
    
    # Check for common prompt injection patterns
    suspicious_patterns = [
        r"ignore previous instructions",
        r"disregard (?:all|previous)",
        r"forget (?:all|previous)",
        r"new prompt",
        r"system prompt",
        r"password|token|api[ _]?key",
        r"execute|eval|system\(",
        r"http[s]?://(?!www\.hopkinsmedicine\.org)",  # Only allow Hopkins website
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, prompt.lower()):
            print(f"Prompt contains suspicious pattern: {pattern}")
            return False
    
    # Sanitized prompt passes basic checks, now send to judge APIs
    starting_message = [
        {
            "role": "user",
            "content": prompt.strip(),
        }
    ]
    response = call_haize_judge(
        [
            "58f2a5e4-d99c-4770-ad83-2ca17d4abf5c",  # patient_name_in_input
            "1ccc3140-c81b-4228-baa0-8a3e8889b2fb",
            "21ac5f3b-0964-47ab-b041-c90840624759",
            "27ba6f76-c78d-4040-927e-3255748d7be7",
    )
    for r in response:
        if r.detected is True:
            print(f"Judge {r.judge_id} detected an issue with the prompt")
            return False
            
    return True



    args = parser.parse_args()

    if args.prompt is not None:
        if not validate_prompt(args.prompt.strip()):
            print("Prompt failed guardrails")
            exit(1)

if __name__ == '__main__':
    main()
