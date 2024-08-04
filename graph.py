from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List, Dict
from langgraph.graph import StateGraph, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

# Define the structure of the agent state using TypedDict
class AgentState(TypedDict):
    agent: str
    initialMessage: str
    responseToUser: str
    lnode: str
    category: str

# Define the structure of category classification
class Category(BaseModel):
    category: str

# Define valid categories
VALID_CATEGORIES = ["policy", "commission", "contest", "ticket", "clarify"]

# Define salesCompAgent class
class salesCompAgent():
    def __init__(self, api_key):
        
        # Initialize the model with given API key
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

        # Build the state graph
        builder = StateGraph(AgentState)
        builder.add_node("classifier", self.initial_classifier)
        builder.add_node("policy", self.policy_agent)
        builder.add_node("commission", self.commission_agent)
        builder.add_node("contest", self.contest_agent)
        builder.add_node("ticket", self.ticket_agent)
        builder.add_node("clarify", self.clarify_agent)

        # Set the entry point and add conditional edges
        builder.set_entry_point("classifier")
        builder.add_conditional_edges("classifier", self.main_router,
                                      {
                                          "policy": "policy",
                                          "commission": "commission",
                                          "contest": "contest",
                                          "ticket": "ticket",
                                          "clarify": "clarify",

                                      })
        
        # Define end points for each node
        builder.add_edge("policy", END)
        builder.add_edge("commission", END)
        builder.add_edge("contest", END)
        builder.add_edge("ticket", END)
        builder.add_edge("clarify", END)

        # Set up in-memory SQLite database for state saving
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(checkpointer = memory)

    # Main router function to direct to the appropriate agent based on the category
    def main_router(self, state: AgentState):
        my_category = state['category']
        if my_category in VALID_CATEGORIES:
            return my_category
        else:
            print(f"unknown category: {my_category}")
            return END

    # Initial classifier function to categorize user messages
    def initial_classifier(self, state: AgentState):
        print("initial_classifier")
        CLASSIFIER_PROMPT = f"""
        You are an expert of customer service in Sales Operations. Please classify the customer
        request as follows:
        1) If the request is a question about sales policies, category is 'policy'
        2) If the request is a question about user's commissions, category is 'commission'
        3) If the request is a question about sales contests, category is 'contest' 
        4) If the request is a question about tickets, category is 'ticket' 
        5) otherwise ask the user to clarify the request, category is 'clarify' 
        """

        # Invoke the model with the classifier prompt
        llm_response = self.model.with_structured_output(Category).invoke([
            SystemMessage(content=CLASSIFIER_PROMPT),
            HumanMessage(content=state['initialMessage']),
        ])

        category = llm_response.category
        print(f"category is {category}")

        # Return the updated state with the category
        return {
            "lnode": "initial_router",
            "responseToUser": "Classifier success",
            "category": category 
        }
    
    # Placeholder function for the policy agent
    def policy_agent(self, state: AgentState):
        print("policy agent")

    # Placeholder function for the commission agent
    def commission_agent(self, state: AgentState):
        print("commission agent")

    # Placeholder function for the contest agent
    def contest_agent(self, state: AgentState):
        print("contest agent")

    # Placeholder function for the ticket agent
    def ticket_agent(self, state: AgentState):
        print("ticket agent")

    # Placeholder function for the clarify agent
    def clarify_agent(self, state: AgentState):
        print("clarify agent")

    
