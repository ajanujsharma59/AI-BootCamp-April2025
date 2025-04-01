import os
import streamlit as st
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()
azure_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# LLM configuration
llm_config = [
    {"model": azure_model_name, "api_key": api_key, 'base_url': azure_endpoint, "api_type": "azure",
     "api_version": "2024-02-01"}
]


# Initialize ChromaDB
def initialize_chromadb_model():
    chroma_client = chromadb.PersistentClient(path="./chroma_csv_db")
    collection = chroma_client.get_or_create_collection(name="autogen_rag_csv")
    return chroma_client, collection


chroma_client, collection = initialize_chromadb_model()

# Initialize Assistant Agent
assistant = AssistantAgent(
    name="Agentic AI Assistant",
    system_message="You are a helpful assistant, Who analyzes the CSV File and always considers Pandas for aggregation.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": llm_config,
    },
)

# Streamlit UI
st.title("Application Tracking System - Assistant")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file
    temp_pdf_path = os.path.join("./", uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Initialize RAG Proxy Agent with uploaded file
    ragproxyagent = RetrieveUserProxyAgent(
        name="Anuj",
        human_input_mode="NEVER",
        retrieve_config={
            "task": "code",
            "docs_path": [temp_pdf_path],
            "custom_text_types": ["pdf"],
            "chunk_token_size": 2000,
            "model": llm_config[0]["model"],
            "client": chroma_client,
            "collection": collection,
            "embedding_model": "all-mpnet-base-v2",
            "get_or_create": True,
            "overwrite": True
        },
        code_execution_config={"use_docker": False}
    )

    # User question input
    user_question = st.text_input("Ask a question about the PDF:")

    if st.button("Submit") and user_question:
        response = ragproxyagent.initiate_chat(
            assistant,
            message=ragproxyagent.message_generator,
            problem=user_question
        )

        response_text = getattr(response, "summary", str(response))
        print("Full Response:", response)
        print("Response Type:", type(response))

        st.write("### Response:")
        st.write(response_text)
