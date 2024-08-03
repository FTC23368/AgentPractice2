import streamlit as st
import os
from graph import salesCompAgent
import random

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGSMITH_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']

DEBUGGING = 0

# This function sets up the chat interface and handles user interactions
def start_chat():
    st.title('Sales Comp Customer Support')
    avatars={"system":"💻🧠","user":"🧑‍💼","assistant":"🎓"}

    # Keeping context of conversations, checks if there is anything in messages array
    # If not, it creates an empty list where all messages will be saved
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Ensuring a unique thread id is maintained for every conversation
    if "thread-id" not in st.session_state:
        st.session_state.thread_id = random.randint(1000, 9999)
    thread_id = st.session_state.thread_id

    # Display previous messages
    for message in st.session_state.messages:
        if message["role"] != "system":
            avatar=avatars[message["role"]]
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    # Handle new user input. Note: Walrus operator serves two functions. It checks if 
    # the user entered any input. If yes, it returns that value.
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=avatars["user"]):
            st.markdown(prompt)

        # Initialize salesCompAgent
        abot=salesCompAgent(st.secrets['OPENAI_API_KEY'])
        thread={"configurable": {"thread_id": thread_id}}

        # Stream responses from the agent
        for s in abot.graph.stream({'initialMessage':prompt},thread):
            if DEBUGGING: 
                print(f"GRAPH RUN: {s}")
                st.write(s)
            for k,v in s.items():
                if DEBUGGING:
                    print(f"Key: {k}, Value: {v}")
                if resp := v.get("responseToUser"):
                    with st.chat_message("assistant", avatar=avatars["assistant"]):
                        st.write(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})

if __name__ == '__main__':
    start_chat()