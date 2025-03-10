import getpass
import os
import asyncio
import argparse
import re
import logging
import sqlite3
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

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


def validate_sql_query(query: str) -> bool:
    """Validate SQL query to prevent harmful operations."""
    disallowed_patterns = [
        r"(?i)\s*INSERT\s+INTO",
        r"(?i)\s*UPDATE\s+",
        r"(?i)\s*DELETE\s+FROM",
        r"(?i)\s*DROP\s+",
        r"(?i)\s*ALTER\s+",
        r"(?i)\s*CREATE\s+",
        r"(?i)\s*TRUNCATE\s+",
        r"(?i)\s*EXEC\s+",
        r"(?i)\s*EXECUTE\s+",
        r"(?i)\s*ATTACH\s+DATABASE",
        r"(?i)\s*DETACH\s+DATABASE",
        r"(?i)\s*PRAGMA"
    ]
    
    for pattern in disallowed_patterns:
        if re.search(pattern, query):
            return False
    return True


_set_env("OPENAI_API_KEY")
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

        goto="supervisor",
    )
class SecureSQLDatabase(SQLDatabase):
    """A secure wrapper around SQLDatabase to validate queries before execution."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def run(self, query: str, fetch: str = "all"):
        """Execute the query if it passes security validation."""
        if not validate_sql_query(query):
            raise ValueError(f"Security violation: Query contains disallowed SQL operations: {query}")
        return super().run(query, fetch=fetch)

def create_database_admin_agent():
    # Create a secure database wrapper with read-only connection
    db_uri = "sqlite:///als_patients.db"
    secure_db = SecureSQLDatabase.from_uri(
        db_uri,
        sample_rows_in_table_info=2,
        connection_arguments={
            "check_same_thread": False, 
            "readonly": True
        }
    )
    toolkit = SQLDatabaseToolkit(db=secure_db, llm=llm)
    tools = toolkit.get_tools()
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt:31156d5f")
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.

    IMPORTANT SECURITY GUIDELINES:
    1. ALWAYS use parameterized queries with placeholders (?) for any variable values.
    2. NEVER concatenate user input directly into SQL queries.
    3. You are NEVER allowed to use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, or any other 
       data modification statements.
    4. You are ONLY allowed to use SELECT statements to read data from the database.
    5. DO NOT attempt to access system tables or execute code outside the database.
    6. LIMIT query results to a reasonable number of rows (use LIMIT clause).
    7. Any queries that would modify the database will be blocked by the system automatically.
    8. If you attempt a disallowed operation, your query will be rejected and you'll need to rewrite it.
    9. Always consider the security implications of your queries.

    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step. Always consider the security implications of your queries.
    Then you should query the schema of the most relevant tables."""
    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
                HumanMessage(content=result["messages"][-1].content, name="database_admin")
            ]
        },
        goto="supervisor",
def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    sql_agent_executor = create_database_admin_agent()
    logging.info(f"Database agent executing with state: {state['messages'][-1].content[:100]}...")
    
    try:
        result = sql_agent_executor.invoke(state)
    except Exception as e:
        logging.error(f"Error in database agent: {str(e)}")
        # Create a fallback result instead of crashing
        return Command(
            update={"messages": [HumanMessage(content=f"I encountered an error while accessing the database: {str(e)}. Please try rephrasing your request.", name="database_admin")]},
            goto="supervisor",
        )
    return Command(
        update={
            "messages": [
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
    if prompt is None:
        return True
        
    logging.info(f"Validating prompt: {prompt[:100]}...")
    
    # Check for SQL injection attempts
    sql_keywords = [
        "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", 
        "UNION", "JOIN", "FROM", "WHERE", "TABLE", "DATABASE", "EXEC", 
        "EXECUTE", "--", ";", "/*", "*/", "INFORMATION_SCHEMA"
    ]
    
    # Check for dangerous patterns
    prompt_upper = prompt.upper()
    for keyword in sql_keywords:
        # Look for SQL keywords with word boundaries
        if re.search(r'\b' + keyword + r'\b', prompt_upper):
            logging.warning(f"Prompt rejected - SQL keyword detected: {keyword}")
            return False
    
    # Check for patient names
    con = sqlite3.connect("als_patients.db")
    cursor = None
    try:
        cursor = con.cursor()
        result = cursor.execute("SELECT name FROM patients ORDER BY name DESC")
        names_list_of_tuples = result.fetchall()
        
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
            logging.warning(f"Prompt rejected - contains patient names: {common_strings}")
            return False
        return True
    except sqlite3.Error as e:
        logging.error(f"Database error during name validation: {str(e)}")
        return False  # Fail closed for security
    finally:
        if cursor: cursor.close()
        con.close()
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("agent_security.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting ALS clinical trial recommendation system")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=False)  # positional argument
    args = parser.parse_args()
    try:
        if args.prompt and not validate_prompt(args.prompt):
            logger.warning("Prompt failed guardrails")
            print("Prompt failed guardrails")
            exit(1)
    except Exception as e:
        logger.error(f"Error in prompt validation: {str(e)}")
        print("Error validating prompt, please check logs")
        exit(2)