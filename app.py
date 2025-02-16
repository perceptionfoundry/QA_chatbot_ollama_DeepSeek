import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Prompt Template
promptTemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that evaluates language models."),
    ("user", "Question: {question}")
])

def generate_response(question, api_key, llm_model, temperature, max_token):
    llm = Ollama(model=llm_model)
   ## llm = ChatOpenAI(model=llm_model,temperature=temperature,max_tokens=max_token,openai_api_key=api_key)
    
    output_parser = StrOutputParser()
    chain = promptTemplate | llm | output_parser
    answer = chain.invoke({"question": question})  # Fix chain execution
    return answer

# st.image("image.png", width=300)
image_path = "image.png"  # Replace with your image path

col1, col2, col3 = st.columns([1, 2, 1])  # Adjust column ratio for centering
with col2:
    st.image(image_path, use_container_width=True)

# Title of the app
st.title("Enhancing Q&A Chatbot with LangChain")

# Sidebar
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Drop Down
llm_model = st.sidebar.selectbox(
    "Select Desired Model",
    ["deepseek-r1:8b","gpt-4o", "gpt-4o-mini", "o1", "o3-mini"]
)

# Adjust response temperature, max_tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
max_token = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main
st.write("Go ahead and ask any question!")
user_input = st.text_input("Question")

if user_input:
    if not api_key:
        st.warning("Hello! How can I help you today?")
    else:
        response = generate_response(user_input, api_key, llm_model, temperature, max_token)
        st.write(response)
else:
    st.write("Please enter a question")