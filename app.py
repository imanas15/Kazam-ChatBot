import streamlit as st
import sys
sys.path.append('C:/Users/HP/Downloads/Final_Year_Project/LangChain.py')
from Kazam_Chat import QnA_Chain

Kazam_about = """We are creating an ecosystem for electric vehicles to thrive- 

1. Integrated network of 50,000+ charge points, easily deployable on any platform
2. Hardware brand-agnostic software to control, assess, analyze & manage money and energy flow for charging hubs or EV fleets
3. Energy Management system to highlight and optimise cost and ensure robust e-fueling for bus depots
4. An array of slow and fast chargers, meant for 2W, 3W, and 4W - to turn any parking spot into a monetised EV charging station"""

st.title("Kazam-Support-Bot")

sidebar = st.sidebar
st.sidebar.title("About")
sidebar.write("")
sidebar.write(Kazam_about)
sidebar.write("")

# Initialize session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Hi.. I am Kazam Support Bot, How can I help you"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Concatenate all user messages into a single query string
    query = " ".join([msg["content"] for msg in st.session_state.messages if msg["role"] == "user"])

    # Get response from RAG_Chain
    response = QnA_Chain(query)  # Assuming response is a string

    # Display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
