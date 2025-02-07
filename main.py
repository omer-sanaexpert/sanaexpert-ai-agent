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
    payload = {
        "action": "getOrderInformation",
        "order_id": order_id
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        auth=HTTPBasicAuth(username, password),
        data=json.dumps(payload)
    )

    print(response.status_code)
    print(response.json())

    return response.json()

@tool
def get_voucher_information() -> Dict[str, Any]:
    """Retrieve current voucher codes and related information.

    Returns:
        Dict[str, Any]: A dictionary containing voucher information.
    """
    print("get_voucher_information")
    payload = {
        "action": "getCurrentShopifyVoucherCodes",
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        auth=HTTPBasicAuth(username, password),
        data=json.dumps(payload)
    )

    print(response.status_code)
    print(response.json())

    return response.json()

@tool
def get_product_information() -> Dict[str, Any]:
    """Retrieve current pricing and url for products.

    Returns:
        Dict[str, Any]: A dictionary containing product pricing , name and url information.
    """
    print("get_product_pricing")
    payload = {
        "action": "getCurrentShopifyPrices",
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        headers=headers,
        auth=HTTPBasicAuth(username, password),
        data=json.dumps(payload)
    )

    print(response.status_code)
    print(response.json())

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
    ("system", """ACT LIKE a friendly SanaExpert customer support agent named Maria and respond on behalf of SanaExpert which is a company who deals in food supplements related to maternity, sports, weight control etc. You will be communicating via chat interface. Please Follow these guidelines:

1. Start with a warm greeting and offer help
2. Handle casual conversation naturally without tools
3. For previous order/shipping queries:
   - First ask for order ID (required) if not provided before.
   - Ask for postal code (required)
   - Only provide information about that specific order.
   - After 3 failed attempts, or If the query is about returning or refund specific product collect name and email and escalate to human agent.
4. If the question is about SanaExpert or its products, policies etc get information using SanaExpertKnowledgebase.
5. For up-to-date product prices and url use get_product_information tool . Remember all prices are in euro and for product restock queries, answer that the product will be back in approx 2 weeks.
6. For voucher related queries use voucher_information tool.
7. Use tools ONLY when specific data is needed.
8. Maintain professional yet approachable tone.
9. Clarify ambiguous requests before acting.
10. Keep your response very brief and concise and ask one thing at a time.
11. Use tools information only in the background and don't tell it to the customer. 
12. In Case you are not sure about answer just ask customer for his name and email if not provided before. and then tell the user that you are escalating the ticket to human representation and then call escalate_to_human tool."""),
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
async def chat(request_data: ChatRequest):
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
def index():
    # Serve the index.html file from the current directory
    return FileResponse("index.html", media_type="text/html")