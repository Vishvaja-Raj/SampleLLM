import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import PyPDF2
import os
import json

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = st.secrets['apikey']

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to save extracted knowledge to a JSON file
def save_knowledge_to_file(pdf_text, file_name="knowledge_store.json"):
    knowledge_store = {"general_info": pdf_text}
    print(knowledge_store)
    with open(file_name, "w") as f:
        json.dump(knowledge_store, f)

# Function to load knowledge from a JSON file
def load_knowledge(file_name="knowledge_store.json"):
    try:
        with open(file_name, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"general_info": ""}

# Prompt template
# Updated Prompt Template
prompt_template = PromptTemplate(
    input_variables=["assistant_name", "knowledge", "history", "current_question", "user_response"],
    template="""
    You are {assistant_name}, a knowledgeable medical assistant bot. 
    You know the following details:
    {knowledge}
    
    Your task is to:
    1. Answer general questions about yourself or unrelated topics when asked.
    2. When the conversation is about health, ask one question at a time to gather details about the user's health issue.
    3. Use the provided conversation history to avoid repeating questions.
    4. If sufficient information about the user's health has been collected, summarize the details and conclude appropriately.

    Conversation History:
    {history}

    Current Question:
    {current_question}

    User's Response:
    {user_response}

    Your Next Response:
    """
)


# Create the LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Streamlit App
st.title("Medical Assistant Bot with Memory")

# Function to extract the assistant's name from the text (assuming it's mentioned in a specific format)
def extract_assistant_name(text):
    # Look for a line starting with "Name:" or similar pattern
    for line in text.splitlines():
        if "Full Name" in line:
            return line.split("Full Name")[1].strip()  # Extract the name after "Name:"
    return "Rohit"  # Default name if not found

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF with knowledge for the assistant", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)  # Extract text from uploaded file
    save_knowledge_to_file(pdf_text)  # Save extracted knowledge to a JSON file
    st.success("Knowledge from the PDF has been saved successfully.")
    
    # Extract the assistant's name and update the session state
    assistant_name = extract_assistant_name(pdf_text)
    st.session_state.assistant_name = assistant_name
    st.success(f"The assistant's name has been set to {assistant_name}.")
    
    # Reload the knowledge after saving
    knowledge = load_knowledge()
    knowledge_context = knowledge.get("general_info", "")  # Update the knowledge context
    st.success("Knowledge updated in the bot.")

# Load knowledge into context
knowledge = load_knowledge()
knowledge_context = knowledge.get("general_info", "")

# Initialize state variables
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = ""
if "current_question" not in st.session_state:
    st.session_state.current_question = "What seems to be the problem?"
if "done" not in st.session_state:
    st.session_state.done = False
if "assistant_name" not in st.session_state:
    st.session_state.assistant_name = "Rohit"  # Default value if not set dynamically

# Function to handle user response
def handle_user_response(user_response):
    # Append the user response to the conversation history
    st.session_state.conversation_history += f"User: {user_response}\n"
    
    # Generate the next question or summary
    next_question_or_summary = llm_chain.run(
        assistant_name=st.session_state.assistant_name,
        knowledge=knowledge_context,
        history=st.session_state.conversation_history,
        current_question=st.session_state.current_question,
        user_response=user_response
    )
    
    # Append the bot's response to the conversation history
    st.session_state.conversation_history += f"Assistant: {next_question_or_summary}\n"
    
    # Check if the conversation is complete
    if "summary" in next_question_or_summary.lower():
        st.session_state.done = True
    else:
        st.session_state.current_question = next_question_or_summary

# Input field for user response
if not st.session_state.done:
    user_response = st.text_input(st.session_state.current_question)
    if user_response:
        handle_user_response(user_response)
        st.experimental_rerun()  # Refresh to show the next question dynamically
else:
    st.write("**Summary of the Information Collected:**")
    st.write(st.session_state.conversation_history)

# Display conversation history (optional for debugging or user transparency)
st.write("### Conversation History:")
st.text_area("Chat History", st.session_state.conversation_history, height=300)
