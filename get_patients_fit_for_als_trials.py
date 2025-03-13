import getpass
import os
import asyncio
import re
from typing import Literal, List
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
from langchain_core.tools import BaseTool


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
        " respond with FINISH."
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


class SecureSQLDatabase:
    """A wrapper around SQLDatabase that adds security measures."""
    
    def __init__(self, db: SQLDatabase):
        self.db = db
        # Patterns for detecting potentially harmful SQL
        self.dangerous_patterns = [
            r'\bDROP\b',
            r'\bDELETE\b',
            r'\bUPDATE\b',
            r'\bINSERT\b',
            r'\bALTER\b',
            r'\bCREATE\b',
            r'\bTRUNCATE\b',
            r'\bEXEC\b',
            r'\bUNION\b.*\bSELECT\b',
            r'--',
            r'\/\*.*\*\/',
            r';.*'  # Prevent multiple statements
        ]
    
    def validate_query(self, query: str) -> bool:
        """Validate a SQL query for potentially harmful operations."""
        query = query.strip().upper()
        
        # Check if query contains dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
                
        # Only allow SELECT statements
        if not query.startswith("SELECT"):
            return False
            
        return True
    
    def run(self, query: str, *args, **kwargs) -> str:
        """Run a SQL query with validation."""
        if not self.validate_query(query):
            return "Error: Query rejected due to security concerns. Only SELECT queries are allowed."
        
        # Execute the query through the underlying database
        return self.db.run(query, *args, **kwargs)


class SecureSQLTool(BaseTool):
    """A wrapper around a SQL tool that adds security measures."""
    
    name: str
    description: str 
    wrapped_tool: BaseTool
    secure_db: SecureSQLDatabase
    
    def _run(self, query: str, *args, **kwargs) -> str:
        """Run the tool with validation."""
        # If this is the query tool, validate the query
        if "run" in self.name.lower() or "query" in self.name.lower():
            if not self.secure_db.validate_query(query):
                return "Error: Query rejected due to security concerns. Only SELECT queries are allowed."
        
        # Run the wrapped tool
        return self.wrapped_tool._run(query, *args, **kwargs)
    
    async def _arun(self, query: str, *args, **kwargs) -> str:
        """Run the tool asynchronously with validation."""
        # If this is the query tool, validate the query
        if "run" in self.name.lower() or "query" in self.name.lower():
            if not self.secure_db.validate_query(query):
                return "Error: Query rejected due to security concerns. Only SELECT queries are allowed."
        
        # Run the wrapped tool asynchronously
        if hasattr(self.wrapped_tool, "_arun"):
            return await self.wrapped_tool._arun(query, *args, **kwargs)
        
        # Fallback to synchronous run
        return self.wrapped_tool._run(query, *args, **kwargs)


def secure_sql_tools(tools: List[BaseTool], secure_db: SecureSQLDatabase) -> List[BaseTool]:
    """Wrap SQL tools with secure versions."""
    secure_tools = []
    
    for tool in tools:
        # Create a secure wrapper around each SQL tool
        secure_tool = SecureSQLTool(
            name=tool.name,
            description=tool.description + "\nNote: Only SELECT statements are allowed for security reasons.",
            wrapped_tool=tool,
            secure_db=secure_db
        )
        secure_tools.append(secure_tool)
    
    return secure_tools


def create_database_admin_agent():
    # Create the database connection
    db = SQLDatabase.from_uri("sqlite:///als_patients.db")
    
    # Create a secure wrapper around the database
    secure_db = SecureSQLDatabase(db)
    
    # Create the toolkit with the original database for schema inspection
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # Get the tools and wrap them with secure versions
    tools = toolkit.get_tools()
    secure_tools = secure_sql_tools(tools, secure_db)

    # Enhanced system message with security warnings
    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    
    SECURITY NOTICE: For security reasons, you are restricted to SELECT statements only. All other SQL operations (INSERT, UPDATE, DELETE, DROP, etc.) will be blocked.
    
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
    
    Use parameterized queries whenever possible by using placeholders like ? or :param instead of directly embedding values in your SQL statements.

    To start you should ALWAYS look at the tables in the database to see what you can query.
    Do NOT skip this step.
    Then you should query the schema of the most relevant tables."""

    # Create the agent with secure tools
    sql_agent_executor = create_react_agent(llm, secure_tools, state_modifier=system_message)
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