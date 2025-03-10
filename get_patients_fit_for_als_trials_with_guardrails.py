import getpass
import os
import asyncio
import re
import argparse
import sqlite3
from typing import Literal
from typing_extensions import TypedDict
from typing import Literal
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, PlayWrightBrowserToolkit
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


def validate_sql_query(query: str) -> tuple[bool, str | None]:
    """Validate that a SQL query does not contain DML statements.
    
    Args:
        query: The SQL query to validate
        
    Returns:
        A tuple with (is_safe, reason). If is_safe is False, reason contains
        the explanation why the query was rejected.
    """
    dangerous_keywords = [
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE",
        "RENAME", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
    ]
    
    # Normalize query for comparison (remove extra whitespace, convert to uppercase)
    normalized_query = " ".join(query.upper().split())
    
    # Check for dangerous SQL operations
    for keyword in dangerous_keywords:
        if keyword in normalized_query.split():
            return False, f"SQL query contains dangerous operation: {keyword}"
    
    # Check for other common SQL injection patterns
    if normalized_query.count(";") > 1 or "--" in normalized_query:
        return False, "SQL query contains potential SQL injection patterns"
    
    return True, None


# set llm and create team members for the lead agent to supervise
llm = ChatOpenAI(model="gpt-4o-mini")
members = ["clinical_researcher", "database_admin"]
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
    return Command(goto=goto)


async def create_restricted_playwright_browser(allowed_urls):
    """Wrap the existing Playwright browser with navigation restrictions.
    
    Args:
        allowed_urls: List of URL prefixes that are allowed for navigation
        
    Returns:
        A function that creates a browser with navigation restrictions
    """
    # Get the original browser creator
    browser_creator = create_async_playwright_browser()
    
    async def create_secured_browser():
        try:
            # Call the original browser creation function
            result = await browser_creator()
            browser, context, page = result
            
            # Set up navigation restriction handler
            async def restrict_navigation(route, request):
                url = request.url
                # Check if the URL is allowed
                if any(url.startswith(allowed_url) for allowed_url in allowed_urls):
                    await route.continue_()  # Allow the navigation
                else:
                    logging.warning(f"Blocked navigation attempt to unauthorized URL: {url}")
                    await route.abort()  # Block the navigation
            
            # Register the handler for all navigation events
            await context.route("**/*", restrict_navigation)
            
            return browser, context, page
        except Exception as e:
            logging.error(f"Error setting up restricted browser: {str(e)}")
            # Fall back to the original browser creator
            return await browser_creator()
    
    return create_secured_browser


def create_clinical_research_agent():
    # Define allowed URLs patterns (only Hopkins Medicine ALS clinical trials)
    allowed_urls = ["https://www.hopkinsmedicine.org/neurology-neurosurgery/clinical-trials/als-clinical-trials"]
    
    # Create a restricted browser that only allows navigation to specified URLs
    async_browser = create_restricted_playwright_browser(allowed_urls)
    clinical_research_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    clinical_research_tools = clinical_research_toolkit.get_tools()

    clinical_research_agent = create_react_agent(
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
    return clinical_research_agent


class SafeSQLDatabase(SQLDatabase):
    """SQLDatabase wrapper with query validation to prevent DML statements."""
    
    @classmethod
    def from_uri(cls, database_uri: str, *args, **kwargs):
        """Create a SafeSQLDatabase from a database URI."""
        # For SQLite, add query parameters to enforce read-only mode
        if database_uri.startswith("sqlite://"):
            if "?" in database_uri:
                database_uri += "&mode=ro"
            else:
                database_uri += "?mode=ro"
        
        return super().from_uri(database_uri, *args, **kwargs)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def run(self, command: str, fetch: str = "all"):
        """Override run method to validate SQL queries before execution."""
        is_safe, reason = validate_sql_query(command)
        if not is_safe:
            print(f"Blocked unsafe SQL query: {reason}")
            return f"Error: {reason}. Only SELECT queries are allowed."
        
        # Additional regex pattern check from the second implementation
        dangerous_patterns = [
            r"\s*insert\s+into", r"\s*update\s+.+\s+set", r"\s*delete\s+from", 
            r"\s*drop\s+", r"\s*alter\s+", r"\s*create\s+", r"\s*truncate\s+",
            r"\s*grant\s+", r"\s*revoke\s+", r"\s*attach\s+", r"\s*detach\s+",
            r"\s*pragma\s+"
        ]
        if any(re.search(pattern, command.lower()) for pattern in dangerous_patterns):
            return "ERROR: This query contains potentially harmful operations and has been blocked for security reasons."
        
        return super().run(command, fetch)


def create_database_admin_agent():
    db = SafeSQLDatabase.from_uri("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm) 
    tools = toolkit.get_tools()

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt:31156d5f")

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    IMPORTANT SAFETY RESTRICTION: You can ONLY use SELECT queries to read data. Any attempt to modify the database will be blocked.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
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


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=False)  # positional argument
    args = parser.parse_args()

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

