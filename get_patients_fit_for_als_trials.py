import getpass
import os
import asyncio
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


class SecureSQLDatabase(SQLDatabase):
    """A more secure version of SQLDatabase with SQL injection protection."""
    
    @classmethod
    def from_uri(cls, database_uri, *args, **kwargs):
        # For SQLite, enforce read-only mode
        if database_uri.startswith("sqlite:") and "?mode=ro" not in database_uri:
            if "?" in database_uri:
                database_uri += "&mode=ro"
            else:
                database_uri += "?mode=ro"
        
        return super().from_uri(database_uri, *args, **kwargs)
    
    def _is_safe_query(self, query: str) -> bool:
        """Check if a SQL query is potentially unsafe."""
        # Check for common SQL injection patterns
        dangerous_patterns = [
            r";\s*DROP\s+", 
            r";\s*DELETE\s+",
            r";\s*INSERT\s+",
            r";\s*UPDATE\s+", 
            r";\s*ALTER\s+",
            r"UNION\s+SELECT",
            r"--\s",
            r"/\*.*\*/",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        
        # Prevent multiple statements
        if query.count(';') > 1:
            return False
            
        # Only allow SELECT statements
        if not query.strip().upper().startswith('SELECT'):
            return False
            
        return True
    
    def run(self, query: str, *args, **kwargs):
        """Run a query with security validation."""
        if not self._is_safe_query(query):
            return "Query rejected for security reasons. Please use only simple SELECT statements."
        
        return super().run(query, *args, **kwargs)


def create_database_admin_agent():
    # Use our secure SQLDatabase subclass with read-only connection
    db = SecureSQLDatabase.from_uri("sqlite:///als_patients.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Enhanced system prompt with explicit security guidelines
    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial.
    
    SECURITY CONSTRAINTS (CRITICAL):
    - ONLY use simple SELECT statements - no other SQL commands are allowed
    - DO NOT use multiple SQL statements (no semicolons except at the end)
    - DO NOT use subqueries with UNION
    - AVOID complex joins that could be used for SQL injection
    - NEVER attempt to modify database structure or data
    - ALWAYS validate your query before executing
    
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