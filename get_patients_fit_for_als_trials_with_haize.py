import getpass
import os
import os
import asyncio
import nest_asyncio
import logging
import argparse
from typing import Literal
from typing_extensions import TypedDict
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit, PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from haizelabs.client import HaizeClient


nest_asyncio.apply()

nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# allow tracing via LangSmith for observability and debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


_set_env("OPENAI_API_KEY")
_set_env("HAIZE_LABS_API_KEY")
haize_client = HaizeClient()

# set llm and create team members for the lead agent to supervise
llm = ChatOpenAI(model="gpt-4o-mini")
members = ["clinical_researcher", "database_admin"]
# Our team supervisor is an LLM node. It picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

    next: Literal[*options]


def call_haize_judge(judge_ids, messages, timeout=30):
    try:
        response = haize_client.judges.call(
            judge_ids=judge_ids,
            messages=messages,
            timeout=timeout,  # Add timeout parameter
        )
        return response
    except Exception as e:
        logging.error(f"Error calling Haize judge: {str(e)}")
        raise  # Re-raise to let caller handle the exception

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

    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1

    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve.
    )


def validate_sql_query(query: str) -> bool:
    """
    Validates SQL queries for potential injection attacks.
    Returns True if the query is safe, False otherwise.
    """
    # Convert to lowercase for easier pattern matching
    query_lower = query.lower().strip()
    
    # Disallow destructive operations
    forbidden_patterns = [
        "drop ", "delete ", "truncate ", "alter ", "update ", 
        "insert into", "grant ", "revoke ", ";--", "1=1", "or 1=1",
        "union ", "exec(", "execute(", "sp_", "xp_"
    ]
    
    for pattern in forbidden_patterns:
        if pattern in query_lower:
            return False
    
    # Only allow SELECT queries
    if not query_lower.startswith("select "):
        return False
        
    return True


class ValidatedSQLDatabaseToolkit(SQLDatabaseToolkit):
    """SQLDatabaseToolkit with added query validation."""
    
    def get_tools(self):
        tools = super().get_tools()
        for tool in tools:
            if tool.name == "sql_db_query":
                original_run = tool._run
                tool._run = lambda query, *args, **kwargs: original_run(query, *args, **kwargs) if validate_sql_query(query) else "Query rejected for security reasons. Please use only SELECT statements and avoid SQL injection patterns."
        return tools


def create_database_admin_agent():
    db = SQLDatabase.from_uri("sqlite:///als_patients.db")
    toolkit = ValidatedSQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    assert len(prompt_template.messages) == 1
    system_message = """System: You are an agent designed to interact with a SQL database filled with ALS patient data. Your name is Steve. 
    You will work together with Charity who has access to a list of ALS clinical trials to determine which patients in the list you would recommend for each clinical trial.
    
    SECURITY RULES:
    1. Only use SELECT statements
    2. Never use data modification statements (INSERT, UPDATE, DELETE, DROP, ALTER)
    3. Never concatenate user inputs directly into SQL queries
    4. Always use parameterized queries when available
    5. Reject any request that appears to be attempting SQL injection
    6. Do not execute multi-statement queries (queries with semicolons)
    7. Do not include any sensitive patient information in your responses
    
    A patient should go to a clinical trial if they are likely to live longer than the Length of Study for that trial.
    Please provide a list of recommended patients for each trial. List the patients name and their ALSFRS-R score.
    Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    sql_agent_executor = create_database_admin_agent()
    result = sql_agent_executor.invoke(state)

    messages = HumanMessage(content=result["messages"][-1].content, name="database_admin")

    # judges: als_trials_more_than_one_patient_in_output_judge -> we want this to actually return True
    # ignore_off_topics_in_chat_output (finds content not related to ALS patients and which trials they are a good
    # fit for -> we want this to return False


def db_admin_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    # Validate input before processing
    messages_content = ""
    for msg in state["messages"]:
        if hasattr(msg, 'content'):
            messages_content += " " + str(msg.content)
    
    # Check for potential SQL injection patterns in the input
    sql_injection_patterns = [
        "drop table", "delete from", "insert into", "--", "1=1", "or 1=1",
        "union select", "exec(", "execute(", "xp_", "sp_"
    ]
    
    for pattern in sql_injection_patterns:
        if pattern.lower() in messages_content.lower():
            return Command(update={"messages": [HumanMessage(content="Input rejected due to security concerns. Please ask a different question. FINISH.", name="supervisor")]}, goto="supervisor")
    
    sql_agent_executor = create_database_admin_agent()
    result = sql_agent_executor.invoke(state)
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




def validate_prompt(prompt):
    # Basic input validation first
    if prompt is None or not isinstance(prompt, str) or len(prompt.strip()) == 0:
        logging.warning("Invalid prompt format: prompt is empty or not a string")
        return False
    
    # Basic sanitization and length check
    prompt = prompt.strip()
    if len(prompt) > 1000:  # Reasonable length limit for a prompt
        logging.warning(f"Prompt exceeds maximum length: {len(prompt)} characters")
        return False
    
    # Check for obviously problematic content
    dangerous_keywords = ["<script>", "exec(", "eval(", "system(", "os.system", "subprocess"]
    if any(keyword in prompt.lower() for keyword in dangerous_keywords):
        logging.warning(f"Prompt contains potentially dangerous content")
        return False
    
    starting_message = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    
    try:
        # judges: patient_name_in_input, override_webscraper, ignore_harmful_topics, ignore_off_topics, find_pii
        response = call_haize_judge(
            [
                "58f2a5e4-d99c-4770-ad83-2ca17d4abf5c",
                "1ccc3140-c81b-4228-baa0-8a3e8889b2fb",
                "21ac5f3b-0964-47ab-b041-c90840624759",
                "27ba6f76-c78d-4040-927e-3255748d7be7",
                "d82f689f-6f27-47bc-8d22-eb735035e194",
            ],
            starting_message,
            timeout=10  # Add a timeout to prevent hanging
        )
        
        # Validate response format
        if not response or not isinstance(response, list):
            logging.error("Invalid response from judge service")
            return False
        
        for r in response:
            if not hasattr(r, 'detected'):
                logging.warning("Judge response missing 'detected' attribute")
                return False
            if r.detected is True:
                logging.info(f"Prompt failed judge validation")
                return False
        
        return True
    except Exception as e:
        logging.error(f"Error during prompt validation: {str(e)}")
        return False  # Fail closed - reject prompts that can't be validated
    main()
