import getpass
import os
import asyncio
from typing import Literal, Callable, Any
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
from urllib.parse import urlparse

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

def is_allowed_url(url: str) -> bool:
    """Only allow navigation to whitelisted clinical trials pages on hopkinsmedicine.org."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("https",):
        return False

    # Allow only main clinical trials page or its subpages under hopkinsmedicine.org with proper path prefix
    hostname = parsed.hostname
    path = parsed.path or ""

    # Allow only the desired path and subpaths (with or without trailing slash)
    allowed_base = "/neurology-neurosurgery/clinical-trials/als-clinical-trials/open-trials"
    if (
        hostname == "www.hopkinsmedicine.org"
        and (
            path == allowed_base
            or path.startswith(allowed_base + "/")
        )
    ):
        return True
    return False

def wrap_playwright_tools_with_url_guard(tools: list) -> list:
    """
    Wrap Playwright browsing tools to restrict navigation to allow-listed domains only.
    If the tool method takes 'url' or 'start_url', enforce the restriction.
    """

    def guarded(tool):
        func = tool.run if hasattr(tool, "run") else tool
        name = getattr(tool, "name", str(tool))

        def guarded_run(*args, **kwargs):
            # Check 'url' or 'start_url' in kwargs, or in single positional argument
            check_url = None
            # Try kwargs
            for key in ["url", "start_url"]:
                if key in kwargs:
                    check_url = kwargs[key]
                    break
            # Else, if single positional arg, use it if param name is 'url' or 'start_url'
            if check_url is None and args:
                # Find out param name from func signature if possible (duck-type)
                # If only one positional argument, likely to be url/start_url
                check_url = args[0] if len(args) == 1 else None
            # Now enforce allow-list
            if check_url is not None:
                if not is_allowed_url(check_url):
                    return {
                        "output": f"ERROR: Access to URL '{check_url}' is denied. Only Hopkins clinical trials pages are permitted."
                    }
            # Forward call
            return func(*args, **kwargs)

        # Maintain API
        class ToolWrapper:
            def __init__(self):
                self.run = guarded_run
                if hasattr(tool, "name"):
                    self.name = tool.name
                if hasattr(tool, "__doc__"):
                    self.__doc__ = tool.__doc__

            # If the tool is callable directly (some are functions), support __call__
            def __call__(self, *args, **kwargs):
                return self.run(*args, **kwargs)
        return ToolWrapper()

    guarded_tools = []
    for tool in tools:
        # Only wrap tools that match browsing/navigate pattern (by name or docstring)
        should_guard = False
        tool_name = getattr(tool, "name", str(tool)).lower()
        for browse_kw in ["navigate", "browse", "goto", "url", "visit"]:
            if browse_kw in tool_name:
                should_guard = True
                break
        # Further, examine docstring for 'url' parameter
        if not should_guard and hasattr(tool, "__doc__") and tool.__doc__:
            if "url" in tool.__doc__ or "start_url" in tool.__doc__:
                should_guard = True

        if should_guard:
            guarded_tools.append(guarded(tool))
        else:
            guarded_tools.append(tool)
    return guarded_tools

def create_clinical_research_agent():
    async_browser = create_async_playwright_browser()
    clinical_research_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

    clinical_research_tools = clinical_research_toolkit.get_tools()

    # Wrap tools with URL guard to prevent SSRF/prompt-injection abuse
    safe_clinical_research_tools = wrap_playwright_tools_with_url_guard(clinical_research_tools)

    clinical_research_agent = create_react_agent(
        llm, tools=safe_clinical_research_tools, state_modifier="You are a ALS clinical researcher. Your name is Charity. Search for ALS clinical trials on https://www.hopkinsmedicine.org/neurology-neurosurgery/clinical-trials/als-clinical-trials/open-trials. Extract text from on the webpage. Let Steve know the Length of Study for each trial."
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

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
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
