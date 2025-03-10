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
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
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
from fastapi.middleware.cors import CORSMiddleware
import time

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
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
index_name = "rag-pinecone-sanaexpertnew"

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
    
    # Find the user_id associated with this thread_id
    user_id = None
    for uid, data in user_conversations.items():
        if data.get("thread_id") == thread_id:
            user_id = uid
            break
    
    # Get conversation history
    conversation_history = ""
    if user_id and user_id in user_conversations:
        conversation_history = "\n".join(user_conversations[user_id].get("history", []))
    
    # Use LLM to generate a summary of the conversation
    summary_prompt = f"""
    Analyze the entire conversation and determine the most appropriate single tag that best represents the main topic or intent. Choose from the following predefined categories: refund, cancel_order, general_information, return_order, subscription, discount or others. Provide only the tag as the response, without any explanation.
    
    
    Conversation:
    {conversation_history} , Tag: 
    """
    
    try:
        # Using the already initialized Anthropic client
        summary_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        # Extract the generated summary
        summary = summary_response.content[0].text

        print("Summary from escalate to human: ", summary)
        
        # Update user details in Zendesk
        if manager.update_user_details(requester_id, ticket_id, email, name , summary):
            # Add the LLM-generated summary as a public comment
            return f"Escalated ticket created for {name} ({email})"
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Fallback to a simple message if LLM fails
        if manager.update_user_details(requester_id, ticket_id, email, name):
            fallback_message = f"Ticket escalated for {name} ({email}). Please review the conversation history."
            if manager.add_public_comment(ticket_id, fallback_message, requester_id):
                return f"Escalated ticket created for {name} ({email})"
    
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
            page_url = configuration.get("page_url", None)
            
            state = {
                **state, 
                "order_id": order_id,
                "thread_id": thread_id,  # Add thread_id to state
                "shipping_url": shipping_url,
                "name": name,
                "email": email,
                "page_url": page_url
            }
            #print("Thread ID from assistant: ", thread_id)
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages, "thread_id": thread_id, "shipping_url": shipping_url , "name": name, "email": email, "page_url": page_url}
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
web_search_tool = TavilySearchResults(k=1, search_engine="google")

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
part_1_tools = [get_order_information_by_orderid,get_order_information_by_email, get_product_information, query_knowledgebase_sanaexpert, escalate_to_human]

# Primary assistant prompt
# Define the primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages([
    ("system", """ 
<persona>
Eres un agente de soporte al cliente amable para SanaExpert, una empresa especializada en suplementos para maternidad, deportes y control de peso. Tu estilo de comunicación es cálido, profesional y eficiente.
</persona>

<responsabilidades_principales>

Identificar las necesidades del cliente
Manejar consultas básicas de manera conversacional
Gestionar consultas sobre pedidos/envíos de forma sistemática
Proporcionar información precisa sobre productos y políticas
Escalar casos complejos a soporte humano
Mantener la conversación breve, concisa y clara
</responsabilidades_principales>
<protocolo_consulta_pedidos>

SIEMPRE recopilar AMBOS datos requeridos en secuencia:
<información_requerida>

Primero: ID de pedido
Segundo: Código postal
</información_requerida>
<reglas_de_validación>

Nunca mencionar ni sugerir ningún código postal
No proceder hasta que el cliente proporcione ambos datos
Si el cliente proporciona solo uno, solicitar el otro
Nunca hacer referencia, sugerir o comparar códigos postales
Validar únicamente la información proporcionada por el cliente
</reglas_de_validación>
<proceso_de_verificación>

Después de recibir el ID de pedido y el código postal:
Usar herramientas para validar la información
Nunca mencionar códigos postales específicos en las respuestas
Si la validación falla: "Noto que hay una discrepancia con la información proporcionada"
</proceso_de_verificación>
<activación_de_escalación>
Tras 3 intentos fallidos de validación:

Solicitar el nombre y el correo electrónico del cliente
Escalar a soporte humano
</activación_de_escalación>
</protocolo_consulta_pedidos>
<protocolo_id_pedido>

Si el cliente solicita su ID de pedido, SIEMPRE recopilar AMBOS datos requeridos en secuencia:
<información_requerida>

Primero: Correo electrónico
Segundo: Código postal
</información_requerida>
<reglas_de_validación>

Nunca mencionar ni sugerir ningún código postal
No proceder hasta que el cliente proporcione ambos datos
Si el cliente proporciona solo uno, solicitar el otro
Nunca hacer referencia, sugerir o comparar códigos postales
Validar únicamente la información proporcionada por el cliente
</reglas_de_validación>
<proceso_de_verificación>

Después de recibir el correo electrónico y el código postal:
Usar herramientas para validar la información
Nunca mencionar códigos postales específicos en las respuestas
Si la validación falla: "Noto que hay una discrepancia con la información proporcionada"
</proceso_de_verificación>
<activación_de_escalación>
Tras 3 intentos fallidos de validación:

Solicitar el nombre del cliente
Escalar a soporte humano
</activación_de_escalación>
</protocolo_id_pedido>
<seguimiento_envío>

Para rastrear envíos: Usar la siguiente URL: {shipping_url}
</seguimiento_envío>
<protocolo_reembolso_cancelación_devolución_modificación>
Para solicitudes de devolución/reembolso o cancelación/modificación de pedidos:

Recopilar el nombre del cliente (obligatorio) y el correo electrónico (obligatorio)
Pregunte el motivo en caso de devolución o reembolso (obligatorio)
Escalar a soporte humano de inmediato
</protocolo_reembolso_cancelación_devolución_modificación>
<protocolo_consulta_vouchers>
Para consultas relacionadas con cupones:

Recopilar el nombre del cliente (obligatorio) y el correo electrónico (obligatorio)
Escalar a soporte humano de inmediato
</protocolo_consulta_vouchers>
<uso_de_herramientas>

SanaExpertKnowledgebase: Para información sobre la empresa/productos/políticas
get_product_information: Para precios actuales (en EUR) y enlaces de productos
escalate_to_human: Para casos complejos que requieran intervención humana. También para devoluciones, reembolsos, cancelaciones o modificaciones de pedidos y solicitudes de escalación
get_order_information_by_orderid: Para obtener detalles del pedido y envío a partir del ID de pedido
get_order_information_by_email: Para obtener detalles del pedido y envío a partir del correo electrónico
</uso_de_herramientas>
<directrices_de_comunicación>

Usar herramientas solo cuando sea necesario
Mantener una comunicación concisa y clara
Hacer una pregunta a la vez
Verificar la comprensión antes de proceder
Mantener el uso de herramientas invisible para los clientes
Nunca revelar ni comparar códigos postales específicos
Para productos agotados: Informar un tiempo de reposición aproximado de 2 semanas
</directrices_de_comunicación>
<protocolo_de_escalación>
Si hay incertidumbre sobre una respuesta:

Se debe recopilar el nombre y el correo electrónico del cliente
Si el cliente no proporciona el nombre y el correo electrónico, solicitarlo
Informar sobre la escalación a soporte humano
Usar la herramienta escalate_to_human
</protocolo_de_escalación>
<manejo_de_conversaciones>
Siempre pasar el thread_id a la herramienta cuando se escale a soporte humano.
ID de hilo actual: {thread_id}.
Nunca compartir el thread_id con el cliente.
</manejo_de_conversaciones>

<current_page_url>
URL de la página actual: {page_url}. No comparta esto con la cliente
</current_page_url>

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



class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for each user")
    message: str = Field(..., description="User message")
    page_url: str = Field(None, description="URL of the page where the chat was initiated")


@app.post("/chat")
async def chat(request_data: ChatRequest, request: Request):
    user_id = request_data.user_id
    user_message = strip_html(request_data.message)
    page_url = request_data.page_url
    print("page url: "+page_url)
    
    if not user_id or not user_message:
        raise HTTPException(status_code=400, detail="Both user_id and message are required")
    
    if user_id not in user_conversations:
        user_conversations[user_id] = {
            "thread_id": str(uuid.uuid4()),
            "history": []
        }
    
    # the problem could be that a user couldnt be able to create multiple tickets.
    thread_id = user_conversations[user_id]["thread_id"]
    print("Thread ID from chat: ", thread_id)
    
    # Check if this thread already has a ticket
    if thread_id not in requests_and_tickets:
        requester_id, ticket_id = manager.create_anonymous_ticket(user_message)
        requests_and_tickets[thread_id] = {
            "requester_id": requester_id,
            "ticket_id": ticket_id
        }
    else:
        requester_id = requests_and_tickets[thread_id]["requester_id"]
        ticket_id = requests_and_tickets[thread_id]["ticket_id"]
    
    # First, add the user message to the ticket
    if not manager.add_public_comment(ticket_id, user_message, requester_id):
        print("Failed to add public comment to ticket")
    else:
        print("Added public comment to ticket")
    
    config = {
        "configurable": {
            "order_id": "",
            "postal_code": "",
            "thread_id": thread_id,
            "email": "",
            "name": "",
            "shipping_url": shipping_url,
            "page_url": page_url
        }
    }
    
    user_conversations[user_id]["history"].append(f"You: {user_message}")
    
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
    
    # Then add the assistant response after getting it
    if last_assistant_response:
        if not manager.add_public_comment(ticket_id, strip_html(last_assistant_response), "32601040249617"):
            print("Failed to add public comment by agent to ticket")
        else:
            print("Added public comment by agent to ticket")
    
    return {"response": last_assistant_response}


@app.get("/")
def index(credentials: HTTPBasicCredentials = Depends(authenticate)):
    return FileResponse("index.html", media_type="text/html")