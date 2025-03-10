import getpass
import os
import asyncio
import re
import logging
import keyring
from typing import Literal, Optional, Dict, Any, List, Tuple, Union
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

 from langchain_core.messages import HumanMessage
 from langgraph.prebuilt import create_react_agent

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# allow tracing via LangSmith for observability and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
def _get_credential(key: str) -> str:
    """
    Securely retrieve or prompt for credentials.
    Uses the system keyring when available, falls back to prompting
    if not previously stored.
    
    Args:
        key: Name of the credential to retrieve
    
    Note: 
        This function handles sensitive information like API keys.
        Values are collected securely and not logged.
        
    Returns:
        str: The retrieved credential
    """
    # Try to get credential from keyring
    credential = keyring.get_password("als_clinical_app", key)
    
    # If not found, prompt user and store in keyring
    if credential is None:
        credential = getpass.getpass(f"{key} (will be stored securely): ")
        
        # Basic validation that a value was provided
        if not credential:
            raise ValueError(f"No value provided for {key}. Operation cancelled.")
            
        keyring.set_password("als_clinical_app", key, credential)
    
    return credential
# set llm and create team members for the lead agent to supervise
api_key = _get_credential("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
members = ["clinical_researcher", "database_admin"]
# Our team supervisor is an LLM node. It picks the next agent to process
# and decides when the work is completed

class Router(TypedDict):
options = members + ["FINISH"]
def validate_router_output(output: Optional[dict], valid_options: list) -> tuple[bool, str]:
    """
    Validates that the router output is one of the expected values.
    
    Args:
        output: The output from the LLM
        valid_options: List of valid routing options
    
    Returns:
        Tuple of (is_valid, value_or_default)
    """
    if not output or "next" not in output:
        logger.warning(f"Invalid router output format: {output}")
        return False, valid_options[0]  # Default to first worker as fallback
    
    if output["next"] not in valid_options:
        logger.warning(f"Unexpected routing value: {output['next']}. Valid options: {valid_options}")
        return False, valid_options[0]  # Default to first worker as fallback
    
    return True, output["next"]

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal[*options]
def supervisor_node(state: MessagesState) -> Command[Literal[*members, "__end__"]]:
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        " respond with FINISH."
    )
    messages = [
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    
    try:
        response = llm.with_structured_output(Router).invoke(messages)
        is_valid, goto = validate_router_output(response, options)
        
        if not is_valid:
            logger.warning("Potentially malicious routing detected, using fallback")
        
        if goto == "FINISH":
            goto = END
    except Exception as e:
        logger.error(f"Error in supervisor routing: {str(e)}")
        goto = options[0]  # Default to first worker as fallback
    
    return Command(goto=goto)

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
    )
class SQLQueryValidator:
    """Validates SQL queries to prevent SQL injection and unauthorized operations."""
    
    FORBIDDEN_PATTERNS = [
        r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE)\b',
        r'\bSYSDBA\b',
        r'\bEXEC\b',
        r'\bEXECUTE\b',
        r'\bATTACH\b',
        r'\bDETACH\b',
        r'\bPRAGMA\b',
        r'--',  # SQL comment
        r';.*?;',  # Multiple statements
        r'\bREPLACE\b',
        r'\bRENAME\b'
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> bool:
        """
        Validates if a SQL query is safe to execute.
        
        Args:
            query (str): The SQL query to validate
            
        Returns:
            bool: True if the query is considered safe, False otherwise
        """
        if not query or not isinstance(query, str):
            logger.warning("Empty or non-string query received")
            return False
            
        # Convert to lowercase for case-insensitive matching
        query_lower = query.strip().lower()
            
        # Check for forbidden patterns
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning(f"Potentially unsafe SQL query detected: {query}")
                return False
                
        # Whitelist approach - only allow SELECT statements
        if not query_lower.startswith("select"):
            logger.warning(f"Non-SELECT query detected: {query}")
            return False
            
        return True


class SafeSQLDatabase(SQLDatabase):
    """A SQL Database with built-in query validation to prevent SQL injection and unsafe operations."""
    
    def run(self, query: str, fetch: str = "all", **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        Safely run a SQL query after validation.
        
        Args:
            query: The SQL query to execute
            fetch: Fetch strategy, one of "all", "one", or "no"
            **kwargs: Additional arguments to pass to the run method
            
        Returns:
            str: Query results or error message
        """
        if not SQLQueryValidator.validate_query(query):
            logger.warning(f"Rejected potentially unsafe query: {query}")
            return "Query rejected for security reasons. Only SELECT statements on approved tables are allowed."
        
        try:
            logger.info(f"Executing validated query: {query}")
            return super().run(query, fetch=fetch, **kwargs)
        except Exception as e:
            logger.error(f"Error executing query: {query}, Error: {str(e)}")
            # Generic error message to avoid information leakage
            return "An error occurred while executing the query. Please try a different query."


def create_database_admin_agent():
    db = SafeSQLDatabase.from_uri("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    assert len(prompt_template.messages) == 1
    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    
    SECURITY GUIDELINES:
    - Only use SELECT statements for querying data
    - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database
    - Do not attempt to access system tables or internal database structures
    - Only query the necessary data needed to answer the question
    - Treat all patient data as confidential and only include relevant information in your responses
    - Avoid using multiple SQL statements separated by semicolons
    - Do not attempt to bypass security measures
    - Any potentially harmful queries will be blocked by the system
    
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step."""
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


async def run_agents():
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

