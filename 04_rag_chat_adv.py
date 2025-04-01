import os
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from dotenv import load_dotenv
import chromadb

load_dotenv()

# Load environment variables
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
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="autogen_rag")
    return chroma_client, collection

chroma_client, collection = initialize_chromadb_model()

# Initialize Assistant Agent
assistant = AssistantAgent(
    name="Agentic AI Assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "cache_seed": 42,
        "config_list": llm_config,
    },
)

# Initialize RAG Proxy Agent with history tracking
ragproxyagent = RetrieveUserProxyAgent(
    name="Anuj",
    human_input_mode="ALWAYS",  # Allow continuous human interaction
    retrieve_config={
        "task": "code",
        "docs_path": [
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
            os.path.join(os.path.abspath(""), "..", "website", "docs"),
        ],
        "custom_text_types": ["mdx"],
        "chunk_token_size": 2000,
        "model": llm_config[0]["model"],
        "client": chroma_client,
        "collection": collection,
        "embedding_model": "all-mpnet-base-v2",
        "get_or_create": True,
    },
    code_execution_config={"use_docker": False}
)

# Continuous Chat Loop
print("Chatbot is ready. Type 'exit' to end the chat.")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Ending chat. Goodbye!")
        break

    ragproxyagent.initiate_chat(
        assistant,
        message=ragproxyagent.message_generator,
        problem=user_input,  # Pass user input as the query
        search_string="spark"  # You can modify this based on user input
    )
