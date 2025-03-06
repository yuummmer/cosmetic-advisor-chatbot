import streamlit as st
import base64
import pandas as pd
from functions import process_multiple_pdfs

st.set_page_config(
    page_title="Cosmetic Advisor Chatbot",
    page_icon="💄✨"
)

# Corrected title function
st.title("Cosmetic Advisor Chatbot")

st.markdown("Welcome! I am your Cosmetic Advisor, here to answer your questions around FDA guidelines for cosmetics. To get started, please upload guidelines from the FDA below.")

# Create an expandable section for uploading files and entering query
with st.expander("Use this section to upload FDA guidelines you'd like help interpreting. I'll read through the document for you and respond from what I understand"):
    uploaded_files = st.file_uploader("Upload PDF invoices:", type=["pdf"], accept_multiple_files=True)

st.title("Cosmetic Advisory Session")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # To store the full chat history (user and assistant messages)

# Display chat messages from history on app rerun (this ensures all chat history is shown)
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to chat about?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Get context from uploaded PDFs if available
    if uploaded_files:
        with st.spinner("Retrieving information from PDFs... Please wait."):

            try:
                # Process the PDFs and get the plain text response
                response = process_multiple_pdfs(uploaded_files, query=prompt)

                # Display assistant's response in chat message container only once
                if response:
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    
                    # Add assistant's response to chat history
                    st.session_state["messages"].append({
                        "role": "assistant", 
                        "content": response
                    })

            except Exception as e:
                st.error(f"An error occurred while processing the PDFs: {str(e)}")

# Add the "Clear History" button to reset chat history
if st.button("Clear Chat History"):
    st.session_state["messages"] = []  # Clear the chat history
    st.rerun()  # This will reset the state and rerun the app

# Add a button to download chat history as CSV
if st.button("Download Chat History to CSV"):
    # Initialize lists to hold user messages and assistant responses
    questions = []
    input_prompts = []
    responses = []

    # Iterate over the chat history and properly pair the user inputs and assistant responses
    user_input = None
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            user_input = message["content"]
        elif message["role"] == "assistant" and user_input:
            # Only append if we have a user input followed by an assistant response
            questions.append(user_input)
            responses.append(message["content"])
            user_input = None  # Reset for the next pair

    # Ensure that the lists have the same length
    if len(questions) == len(responses):
        # Create a DataFrame from the data
        data = {
            'questions': questions,
            'responses': responses
        }
        chat_df = pd.DataFrame(data)

        # Convert the DataFrame to CSV format
        csv = chat_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Encode the CSV to base64

        # Create a download link for the CSV file
        href = f'<a href="data:file/csv;base64,{b64}" download="chat_history.csv">Download Chat History</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.error("The number of questions and responses are mismatched. Please check the chat history.")