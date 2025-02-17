from fastapi import FastAPI, Depends, HTTPException, Body, Request
import geoip2.database
from geoip2.errors import AddressNotFoundError
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
from typing import Annotated, Optional, TypedDict, List  # Import for State definition
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
from starlette.middleware.sessions import SessionMiddleware
from starlette.datastructures import Headers, QueryParams


from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends

from datetime import datetime, timedelta

from anthropic import Anthropic, Client
from dotenv import load_dotenv

from zendesklib import ZendeskTicketManager
from user_agents import parse
import re
from html import unescape
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())

load_dotenv() 



manager = ZendeskTicketManager()
# Cache dictionary to store API responses
api_cache = {
    "get_order_information_by_email": {},
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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != AUTH_USERNAME or credentials.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)



def strip_html(content: str) -> str:
    """Removes code blocks, HTML tags, and unnecessary whitespace from the given content."""
    
    # Remove code blocks (content between triple backticks)
    content = re.sub(r'```[\s\S]*?```', '', content)
    
    # Remove style attributes
    content = re.sub(r'style="[^"]*"', '', content)
    
    # Remove HTML comments
    content = re.sub(r'<!--[\s\S]*?-->', '', content)
    
    # Remove script tags and their content
    content = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', content, flags=re.IGNORECASE)
    
    # Remove style tags and their content
    content = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', content, flags=re.IGNORECASE)
    
    # Replace common block elements with newlines
    content = re.sub(r'</(div|p|h[1-6]|table|tr|li)>', '\n', content, flags=re.IGNORECASE)
    
    # Replace <br> tags with newlines
    content = re.sub(r'<br[^>]*>', '\n', content, flags=re.IGNORECASE)
    
    # Remove all remaining HTML tags
    content = re.sub(r'<[^>]+>', '', content)
    
    # Decode HTML entities
    content = unescape(content)  # Converts &nbsp;, &amp;, &lt;, &gt;, etc.

    # Clean up whitespace
    content = re.sub(r'\n\s*\n', '\n', content)  # Remove multiple empty lines
    content = content.strip()  # Trim start and end
    content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces and tabs
    
    # Normalize lines
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    return '\n'.join(lines)





def count_tokens(text: str) -> int:
    """Count the number of tokens in a given text."""
    # Count tokens before creating a message
    print(text)
    count = client.beta.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": text}],
    )
    return count



# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(SessionMiddleware, secret_key="your-secret-keykshdfbdsjkfh")
print("welcome")

# In-memory storage for user conversations
user_conversations = {}
# In-memory storage for request and ticket IDs
requests_and_tickets = {}

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



# Load the multilingual-e5-small model
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
    thread_id: str | None
    shipping_url: str | None
    name : str | None
    email : str | None
    order_id: str | None
    postal_code: str | None

@tool
def get_order_information_by_orderid(order_id: str) -> Dict[str, Any]:
    """Retrieve order and shipping details by order ID.

    Args:
        order_id (str): The unique identifier for the order.

    Returns:
        Dict[str, Any]: A dictionary containing order details, including shipping information.
    """
    print("get_order_information_by_orderid")
    print("order id : ",order_id)

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
def get_order_information_by_email(email: str) -> Dict[str, Any]:
    """Retrieve order and shipping details of the last order by email.

    Args:
        email (str): The email of the customer.

    Returns:
        Dict[str, Any]: A dictionary containing order details, including shipping information.
    """
    print("get_order_information_by_email")
    print("email: ",email)

    # Check if the order data is cached and still valid
    if email in api_cache["get_order_information_by_email"]:
        cached_entry = api_cache["get_order_information"][email]
        if is_cache_valid(cached_entry["timestamp"]):
            print("Returning cached order info")
            return cached_entry["data"]

    # If not cached or expired, call API
    payload = {
        "action": "getOrderInformation",
        "mail_address": email
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, auth=HTTPBasicAuth(username, password), data=json.dumps(payload))

    # Store response in cache
    api_cache["get_order_information_by_email"][email] = {"data": response.json(), "timestamp": datetime.now()}
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

    

@tool
def escalate_to_human(name: str, email: str, thread_id: str) -> str:
    """Escalate the conversation to a human agent.

    Args:
        name (str): The name of the person requesting escalation.
        email (str): The email address of the person requesting escalation.
        thread_id (str): The thread ID associated with the user's session.

    Returns:
        str: A confirmation message indicating the ticket has been escalated.
    """
    print("escalate_to_human", name, email)
    if not name or not email:
        return "Please provide both your name and email to escalate the ticket."
    print("thread id "+thread_id)
    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    #TODO: Create a ticket in Zendesk
    if manager.update_user_details(requester_id,ticket_id, email, name):
        return f"Escalated ticket created for {name} ({email})"
    else:
        return "Something went wrong. Please contact support@sanaexpert.com"

@tool
def query_knowledgebase_sanaexpert(q: str) -> str:
    """Query the SanaExpert knowledge base for product information, return policies, shipment policies, and general information.

    Args:
        q (str): The query string to search in the knowledge base.

    Returns:
        str: A concatenated string of the top 5 matching results from the knowledge base.
    """
    print("query_knowledgebase_sanaexpert")
    query_embedding = embedding_model.encode([q])[0].tolist()
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=5,
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
        [RunnableLambda(handle_tool_error)], 
        exception_key="error"
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
            thread_id = configuration.get("thread_id", None)  # Get thread_id from config
            shipping_url = configuration.get("shipping_url", None)
            
            state = {
                **state, 
                "order_id": order_id,
                "thread_id": thread_id,  # Add thread_id to state
                "shipping_url": shipping_url,
                "name": name,
                "email": email
            }
            #print("Thread ID from assistant: ", thread_id)
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages, "thread_id": thread_id, "shipping_url": shipping_url , "name": name, "email": email}
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
part_1_tools = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human, get_voucher_information]

# Primary assistant prompt
# Define the primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", """<persona>
    You are a friendly customer support agent for SanaExpert, a company specializing in maternity, sports, and weight control supplements. Your communication style is warm, professional, and efficient.
</persona>

<core_responsibilities>
- Identify the customer needs
- Handle basic inquiries conversationally
- Manage order/shipping queries systematically
- Provide accurate product and policy information
- Escalate complex cases to human support
- Keep the conversation short, consise and clear
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
     
<order_id_protocol>
1. If user ask for order id ALWAYS collect BOTH required pieces of information in sequence:
   <required_info>
   - First: email
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
   - After receiving both email and postal code:
     * Use tools to validate the information
     * Never mention specific postal codes in responses
     * If validation fails: "I notice there's a mismatch with the provided information"
   </verification_process>

   <escalation_trigger>
   After 3 failed validation attempts:
   1. Request customer name
   2. Escalate to human support
   </escalation_trigger>
</order_id_protocol>

<shippment_url>
- For shipment tracking: Use the following URL: {shipping_url}
</shippment_url>

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
- get_order_information_by_orderid: For order and shipping details from order ID
- get_order_information_by_email: For order and shipping details from order ID
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
1. Must Collect customer name and email
2. if user doesnt provide email and name, ask for it
2. Inform about escalation to human support
3. Use escalate_to_human tool
</escalation_protocol>

<thread_handling>
Always pass the thread_id to tool when escalating to human support.
Current thread ID: {thread_id}.
Never share the thread_id with the customer.
</thread_handling>
     
<response_format>
     must be a valid html and basic css. brand color is #0d8500 , link target new tab. use buttons for links.
</response_format>
     
     
     """),
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

class BrowserInfo(BaseModel):
    browser_family: str
    browser_version: Optional[str]
    os_family: str
    os_version: Optional[str]
    device_family: str
    device_brand: Optional[str]
    device_model: Optional[str]
    is_mobile: bool
    is_tablet: bool
    is_desktop: bool
    is_bot: bool
    raw_user_agent: str

class LocationInfo(BaseModel):
    country_code: Optional[str]
    country_name: Optional[str]
    city: Optional[str]
    postal_code: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[str]
    continent: Optional[str]
    subdivision: Optional[str]
    accuracy_radius: Optional[int]

class RequestInfo(BaseModel):
    # Previous request fields...
    method: str
    url: str
    base_url: str
    path: str
    headers: Dict[str, str]
    client_host: Optional[str]
    
    # New fields for browser and location
    browser_info: Optional[BrowserInfo]
    location_info: Optional[LocationInfo]

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for each user")
    message: str = Field(..., description="User message")
    request_info: Optional[RequestInfo] = None

def parse_browser_info(user_agent_string: str) -> BrowserInfo:
    """Parse user agent string to extract detailed browser information"""
    if not user_agent_string:
        return None
    
    # Parse the user agent string
    user_agent = parse(user_agent_string)
    
    return BrowserInfo(
        browser_family=user_agent.browser.family,
        browser_version=str(user_agent.browser.version_string),
        os_family=user_agent.os.family,
        os_version=str(user_agent.os.version_string),
        device_family=user_agent.device.family,
        device_brand=user_agent.device.brand,
        device_model=user_agent.device.model,
        is_mobile=user_agent.is_mobile,
        is_tablet=user_agent.is_tablet,
        is_desktop=user_agent.is_pc,
        is_bot=user_agent.is_bot,
        raw_user_agent=user_agent_string
    )

def get_location_info(ip_address: str) -> LocationInfo:
    """Get location information from IP address using MaxMind GeoIP2 database"""
    try:
        # Initialize the GeoIP2 reader with the MaxMind database
        # You need to download the GeoIP2 database from MaxMind and specify the path
        with geoip2.database.Reader('GeoLite2-City.mmdb') as reader:
            response = reader.city(ip_address)
            
            return LocationInfo(
                country_code=response.country.iso_code,
                country_name=response.country.name,
                city=response.city.name,
                postal_code=response.postal.code,
                latitude=response.location.latitude,
                longitude=response.location.longitude,
                timezone=response.location.time_zone,
                continent=response.continent.name,
                subdivision=response.subdivisions.most_specific.name if response.subdivisions else None,
                accuracy_radius=response.location.accuracy_radius
            )
    except (AddressNotFoundError, FileNotFoundError):
        return None

async def extract_request_info(request: Request) -> RequestInfo:
    """Extract all available information from the request object"""
    # Get headers and other basic info
    headers_dict = dict(request.headers)
    client = request.client
    client_host = client.host if client else None
    
    # Parse browser information
    user_agent_string = headers_dict.get('user-agent')
    browser_info = parse_browser_info(user_agent_string) if user_agent_string else None
    
    # Get location information
    location_info = get_location_info(client_host) if client_host else None
    
    return RequestInfo(
        method=request.method,
        url=str(request.url),
        base_url=str(request.base_url),
        path=request.url.path,
        headers=headers_dict,
        client_host=client_host,
        browser_info=browser_info,
        location_info=location_info
    )

@app.post("/chat")
async def chat(request_data: ChatRequest, request: Request, credentials: HTTPBasicCredentials = Depends(authenticate)):
    user_id = request_data.user_id
    user_message = strip_html(request_data.message)
    print(user_message)

    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="Both user_id and message are required")

    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": []
        }
        requester_id, ticket_id = manager.create_anonymous_ticket(user_message)
        requests_and_tickets[user_conversations[user_id]["thread_id"]] = {
            "requester_id": requester_id,
            "ticket_id": ticket_id
        }

    thread_id = user_conversations[user_id]["thread_id"]
    print("Thread ID from chat: ", thread_id)
    config = {
        "configurable": {
            "order_id": "",
            "postal_code": "",
            "thread_id": thread_id,
            "email": "",
            "name": "",
            "shipping_url": shipping_url
        }
    }

    user_conversations[user_id]["history"].append(f"\U0001F9D1\u200D\U0001F4BB You: {user_message}")
    input_tokens = count_tokens(user_message)
    
    # Initialize a set to track printed events
    printed_events = set()

    try:
        events = part_1_graph.stream(
            {"messages": [("user", (user_message))]}, config, stream_mode="values"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch AI response: {str(e)}")

    last_assistant_response = ""
    raw_events = list(events)
    for event in raw_events:
        # Print each event
        _print_event(event, printed_events)
        
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
    
    output_tokens = count_tokens(last_assistant_response)

    print("Input tokens: ", input_tokens)
    print("Output tokens: ", output_tokens)

    requester_id = requests_and_tickets[thread_id]["requester_id"]
    ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    if not manager.add_public_comment(ticket_id, (user_message), requester_id):
        print("Failed to add public comment to ticket")
    else:
        print("Added public comment to ticket")
    if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
        print("Failed to add public comment by agent to ticket")
    else:
        print("Added public comment by agent to ticket")

    return {"response": last_assistant_response}


@app.get("/")
def index(credentials: HTTPBasicCredentials = Depends(authenticate)):
    return FileResponse("index.html", media_type="text/html")