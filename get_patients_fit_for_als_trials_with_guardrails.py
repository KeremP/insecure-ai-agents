import getpass
import os
import asyncio
import argparse
import sqlite3
import re
import logging
from typing import Literal, List, Any, Dict, Optional
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.agent_toolkits import SQLDatabaseToolkit, PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# allow tracing via LangSmith for observability and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
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


    )


class SafeSQLDatabase(SQLDatabase):
    """A wrapper around SQLDatabase that enforces read-only operations and query validation."""
    
    FORBIDDEN_PATTERNS = [
        # Match common DML operations
        r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE)\b',
        # Match potentially dangerous operations
        r'\bATTACH\b',
        r'--',  # SQL comment that might be used to terminate valid queries
        r';.*;', # Multiple queries in one statement
    ]
    
    @classmethod
    def from_uri(cls, database_uri: str, **kwargs):
        # Force read-only mode for SQLite connections
        if database_uri.startswith('sqlite:'):
            if '?' in database_uri:
                database_uri += '&mode=ro'
            else:
                database_uri += '?mode=ro'
                
        # Add engine arguments to ensure read-only
        engine_kwargs = kwargs.get('engine_kwargs', {})
        engine_kwargs['connect_args'] = engine_kwargs.get('connect_args', {})
        engine_kwargs['connect_args']['readonly'] = True
        kwargs['engine_kwargs'] = engine_kwargs
        
        return super().from_uri(database_uri, **kwargs)
    
    def validate_query(self, query: str) -> bool:
        """Validate that a query is safe to execute."""
        # Check for forbidden patterns
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        
        # Ensure query is a SELECT statement
        if not re.match(r'^\s*SELECT\b', query, re.IGNORECASE):
            return False
            
        return True
    
    def run(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Run a query after validating it."""
        if not self.validate_query(query):
            raise ValueError("Query validation failed: only safe SELECT operations are allowed")
            
        return super().run(query, **kwargs)


def is_safe_sql_query(query: str) -> bool:
    """
    Validate if a SQL query is safe to execute.
    Returns False for dangerous operations like DROP, DELETE, UPDATE, etc.
    """
    if not query:
        return False
        
    # Convert to lowercase for case-insensitive matching
    query_lower = query.lower().strip()
    
    # Block any data manipulation or definition statements
    dangerous_keywords = [
        "drop", "delete", "update", "insert", "alter", "create", 
        "truncate", "replace", "vacuum", "into", "pragma", "attach"
    ]
    
    # Check if query contains any dangerous keywords
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', query_lower):
            logger.warning(f"Potentially unsafe SQL query detected: {query}")
            return False
            
    # Ensure it's a SELECT statement (legitimate read operation)
    if not re.search(r'^\s*select\b', query_lower):
        logger.warning(f"Non-SELECT SQL query detected: {query}")
        return False
        
    return True


class SafeSQLDatabaseToolkit(SQLDatabaseToolkit):
    """Enhanced SQLDatabaseToolkit with query validation"""
    
    def get_tools(self):
        tools = super().get_tools()
        
        # Wrap the query tool with validation
        original_run_query_tool = next((tool for tool in tools if tool.name == "sql_db_query"), None)
        
        if original_run_query_tool:
            original_run = original_run_query_tool._run
            
            # Replace the _run method with a safe version
            def safe_run(query: str) -> str:
                if not is_safe_sql_query(query):
                    return "Query rejected: Contains potentially unsafe operations. Please use only SELECT statements."
                return original_run(query)
                
            original_run_query_tool._run = safe_run
        
        return tools


def create_database_admin_agent():
    db = SafeSQLDatabase.from_uri("sqlite:///als_patients.db")

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt:31156d5f")
    assert len(prompt_template.messages) == 1
    You can order the results by a relevant column to return the most interesting examples in the database.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    ONLY use SELECT statements. Any other SQL operations (INSERT, UPDATE, DELETE, DROP, etc.) will be blocked by the security system.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    If your query is rejected for security reasons, simplify it to use only basic SELECT statements.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. These will be blocked.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Only retrieve the data you need to answer the question.
    Then you should query the schema of the most relevant tables."""

    toolkit = SafeSQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    sql_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
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
                            if prompt is None else prompt,
                    )
                ]
            },
            subgraphs=True,
    ):
        print(s)


def validate_prompt(prompt):
    if prompt is None:
        return True
        
    # Prevent SQL injection patterns in the prompt
    sql_patterns = [
        r'[\'"\\\;]',                    # SQL special characters
        r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\b', # SQL keywords
        r'--',                          # SQL comments
        r'/\*.*?\*/'                    # SQL block comments
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            return False
    
    # Also keep the original check for patient names
    try:
        con = sqlite3.connect("als_patients.db", mode="ro")
        con.execute("PRAGMA query_only = ON;")  # Ensure read-only mode
        cursor = con.cursor()
        result = cursor.execute("SELECT name FROM patients ORDER BY name DESC")
        names_list_of_tuples = result.fetchall()
        cursor.close()
        con.close()
        list_of_names = []
        for name in names_list_of_tuples:
            full_name = name[0].split(" ")
            if len(full_name) >= 2:
                list_of_names.extend(full_name)
                
        words_in_prompt = prompt.split(" ")
        common_strings = set(list_of_names) & set(words_in_prompt)
    except Exception:
        return False
        
    if common_strings:
        return False
    else:

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=False)
    args = parser.parse_args()

    if args.prompt is not None and not isinstance(args.prompt, str):
        print("Prompt must be a string")
        exit(1)
        
    if args.prompt is not None and not validate_prompt(args.prompt):
        print("Prompt failed guardrails")
        exit(1)

    loop.close()


if __name__ == '__main__':