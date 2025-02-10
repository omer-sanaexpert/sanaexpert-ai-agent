from fastapi import FastAPI, Depends, HTTPException, Body
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
from typing import Annotated, TypedDict, List  # Import for State definition
from langgraph.graph.message import AnyMessage, add_messages  # Import for State definition
import uuid
import os
from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool  # Import the @tool decorator
import requests
from requests.auth import HTTPBasicAuth
import json
from bs4 import BeautifulSoup
import re
from langchain_core.output_parsers import StrOutputParser
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List

from fastapi.responses import JSONResponse


from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends

from datetime import datetime, timedelta

# Cache dictionary to store API responses
api_cache = {
    "get_order_information": {},
    "get_voucher_information": {"data": None, "timestamp": None},
    "get_product_information": {"data": None, "timestamp": None}
}

CACHE_EXPIRY_HOURS = 24  # Set cache expiration time to 24 hours

def is_cache_valid(timestamp):
    """Check if the cached data is still valid."""
    return timestamp and datetime.now() - timestamp < timedelta(hours=CACHE_EXPIRY_HOURS)


# Security setup
security = HTTPBasic()
AUTH_USERNAME = os.getenv("API_USERNAME", "sanaexpert")  # Set these in your .env file
AUTH_PASSWORD = os.getenv("API_PASSWORD", "San@Xpert997755")

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != AUTH_USERNAME or credentials.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# In-memory storage for user conversations
user_conversations = {}

# Endpoint URL
url = os.environ.get("SCL_URL")
username = os.environ.get("SCL_USERNAME")
password = os.environ.get("SCL_PASSWORD")
shipping_url = os.environ.get("SHIPMENT_TRACKING_URL")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "rag-pinecone-sanaexpert"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)



# Load the LaBSE model
embedding_model = None
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")



LOG_FILE = "chat_logs.json"

def save_log(user_id, user_message, assistant_response):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "user_message": user_message,
        "assistant_response": assistant_response
    }

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(log_entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


# generate and store the embeddings for the knowledgebase

# Define the State class
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

@tool
def get_order_information(order_id: str) -> Dict[str, Any]:
    """Retrieve order and shipping details by order ID.

    Args:
        order_id (str): The unique identifier for the order.

    Returns:
        Dict[str, Any]: A dictionary containing order details, including shipping information.
    """
    print("get_order_information")

    # Check if the order data is cached and still valid
    if order_id in api_cache["get_order_information"]:
        cached_entry = api_cache["get_order_information"][order_id]
        if is_cache_valid(cached_entry["timestamp"]):
            print("Returning cached order info")
            return cached_entry["data"]

    # If not cached or expired, call API
    payload = {
        "action": "getOrderInformation",
        "order_id": order_id
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_order_information"][order_id] = {"data": response.json(), "timestamp": datetime.now()}
    return response.json()

@tool
def get_voucher_information() -> Dict[str, Any]:
    """Retrieve current voucher codes and related information.

    Returns:
        Dict[str, Any]: A dictionary containing voucher information.
    """
    print("get_voucher_information")

    if is_cache_valid(api_cache["get_voucher_information"]["timestamp"]):
        print("Returning cached voucher info")
        return api_cache["get_voucher_information"]["data"]

    # Fetch from API if cache is expired
    payload = {"action": "getCurrentShopifyVoucherCodes"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_voucher_information"] = {"data": response.json(), "timestamp": datetime.now()}
    return response.json()

@tool
def get_product_information() -> Dict[str, Any]:
    """Retrieve current pricing and url for products.

    Returns:
        Dict[str, Any]: A dictionary containing product pricing , name and url information.
    """
    print("get_product_pricing")

    if is_cache_valid(api_cache["get_product_information"]["timestamp"]):
        print("Returning cached product info")
        return api_cache["get_product_information"]["data"]

    # Fetch from API if cache is expired
    payload = {"action": "getCurrentShopifyPrices"}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_product_information"] = {"data": response.json(), "timestamp": datetime.now()}
    return response.json()

# @tool
# def get_product_url() -> str:
#     """Retrieve the URL for a given product.

#     Returns:
#         str: The URL of the product.
#     """
#     print("get_product_url")
    
#     df = pd.read_csv('product.csv', delimiter=';', encoding='unicode_escape')
#     df.drop(['sku', 'product_description'], axis=1, inplace=True)

#     print(df.to_string())


#     return df.to_string() + "\n\n ."
    

@tool
def escalate_to_human(name: str, email: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        name (str): The name of the person requesting escalation.
        email (str): The email address of the person requesting escalation.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("escalate_to_human", name, email)
    if not name or not email:
        return "Please provide both your name and email to escalate the ticket."
    return f"Escalated ticket created for {name} ({email})"

@tool
def query_knowledgebase_sanaexpert(q: str) -> str:
    """Query the SanaExpert knowledge base for product information, return policies, shipment policies, and general information.

    Args:
        q (str): The query string to search in the knowledge base.

    Returns:
        str: A concatenated string of the top 3 matching results from the knowledge base.
    """
    print("query_knowledgebase_sanaexpert")
    query_embedding = embedding_model.encode([q])[0].tolist()
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True
    )
    return "\n\n".join([match.metadata["text"] for match in results.matches])

def handle_tool_error(state) -> dict:
    print("handle_tool_error" , state.get("error"))
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

# Define the Assistant class
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            order_id = configuration.get("order_id", None)
            name = configuration.get("name", None)
            email = configuration.get("email", None)
            state = {**state, "user_info": order_id}
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Question rewriter
websystem = """You are a question re-writer that converts an input question to a better version optimized for web search."""
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", websystem),
    ("human", "Here is the initial question:\n\n{question}\nFormulate an improved question."),
])
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=1)

#llm = ChatGroq(model="llama3-70b-8192", temperature=1)
question_rewriter = re_write_prompt | llm | StrOutputParser()

# Web search tool
web_search_tool = TavilySearchResults(k=3, search_engine="google")

@tool
def web_search(query: str) -> str:
    """
    Perform a web search based on the given query.

    Args:
        query (str): The query for the web search.

    Returns:
        str: A string containing the search results.
    """
    print("web search")
    rewritten_query = question_rewriter.invoke({"question": query})
    print(rewritten_query)
    
    # Perform web search
    docs = web_search_tool.invoke({"query": rewritten_query}) or []

    print(docs[0]['content'] if docs else "No results found.")

    return docs[0]['content'] if docs else "No results found."


# Tools list
part_1_tools = [get_order_information, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human, get_voucher_information]

# Primary assistant prompt
# Define the primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", """<persona>
    You are Maria, a friendly customer support agent for SanaExpert, a company specializing in maternity, sports, and weight control supplements. Your communication style is warm, professional, and efficient.
</persona>

<core_responsibilities>
- Greet customers warmly and identify their needs
- Handle basic inquiries conversationally
- Manage order/shipping queries systematically
- Provide accurate product and policy information
</core_responsibilities>

<order_query_protocol>
1. ALWAYS collect BOTH required pieces of information in sequence:
   <required_info>
   - First: Order ID
   - Second: Postal code
   </required_info>

   <validation_rules>
   - Never mention or suggest any postal code
   - Do not proceed until both pieces are provided by customer
   - If customer provides only one, ask for the other
   - Never reference, suggest, or compare postal codes
   - Only validate what customer provides
   </validation_rules>

   <verification_process>
   - After receiving both Order ID and postal code:
     * Use tools to validate the information
     * Never mention specific postal codes in responses
     * If validation fails: "I notice there's a mismatch with the provided information"
   </verification_process>

   <escalation_trigger>
   After 3 failed validation attempts:
   1. Request customer name and email
   2. Escalate to human support
   </escalation_trigger>
</order_query_protocol>

<refund_protocol>
For return/refund requests:
1. Collect customer name and email
2. Escalate to human support immediately
</refund_protocol>

<tool_usage>
- SanaExpertKnowledgebase: For company/product/policy information
- get_product_information: For current prices (in EUR) and URLs
- voucher_information: For promotional code details
- escalate_to_human: For complex cases requiring human intervention
</tool_usage>

<communication_guidelines>
- Maintain concise, clear communication
- Ask one question at a time
- Verify understanding before proceeding
- Keep tool usage invisible to customers
- Never reveal or compare specific postal codes
- For out-of-stock items: Inform 2-week approximate restock time
</communication_guidelines>

<escalation_protocol>
If uncertain about any response:
1. Collect customer name and email
2. Inform about escalation to human support
3. Use escalate_to_human tool
</escalation_protocol>"""),
    ("placeholder", "{messages}"),
]).partial(time=datetime.now)

# Build assistant runnable
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

# Build graph
builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

# Chat endpoint
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for each user")
    message: str = Field(..., description="User message")

@app.post("/chat")
async def chat(request_data: ChatRequest, credentials: HTTPBasicCredentials = Depends(authenticate)):
    user_id = request_data.user_id
    user_message = request_data.message

    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="Both user_id and message are required")

    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": []
        }

    thread_id = user_conversations[user_id]["thread_id"]
    config = {
        "configurable": {
            "order_id": "",
            "thread_id": thread_id,
        }
    }

    user_conversations[user_id]["history"].append(f"\U0001F9D1\u200D\U0001F4BB You: {user_message}")

    try:
        events = part_1_graph.stream(
            {"messages": [("user", user_message)]}, config, stream_mode="values"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")

    last_assistant_response = ""
    raw_events = list(events)
    for event in raw_events:
        if "messages" in event:
            for message in event["messages"]:
                if hasattr(message, "content") and "AIMessage" in str(type(message)):
                    content = message.content
                    if isinstance(content, dict) and "text" in content:
                        content = content["text"]
                    elif isinstance(content, list):
                        content = " ".join(str(part) for part in content)
                    elif isinstance(content, str):
                        last_assistant_response = content

    return {"response": last_assistant_response}


@app.get("/")
def index(credentials: HTTPBasicCredentials = Depends(authenticate)):
    return FileResponse("index.html", media_type="text/html")