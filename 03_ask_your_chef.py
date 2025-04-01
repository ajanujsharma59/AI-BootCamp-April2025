import os
from dotenv import load_dotenv
from autogen import ConversableAgent, GroupChat, GroupChatManager

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
azure_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
max_turns = 2

llm_config = {
    "config_list": [
        {
            "model": azure_model_name,
            "api_key": api_key,
            "base_url": azure_endpoint,
            "api_type": "azure",
            "api_version": "2024-02-01"
        }
    ]
}

# Define Cuisine Chefs
chefs = [
    ConversableAgent(
        name="Italian_Chef",
        system_message="Expert in Italian cuisine, providing details about pasta, pizza, and traditional Italian dishes.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    ),
    ConversableAgent(
        name="North_Indian_Chef",
        system_message="Expert in North Indian cuisine, knowledgeable about curries, tandoori dishes, and rich gravies.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    ),
    ConversableAgent(
        name="South_Indian_Chef",
        system_message="Specialist in South Indian cuisine, including dosas, idlis, sambars, and coconut-based dishes.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    ),
    ConversableAgent(
        name="Bengali_Chef",
        system_message="Specialist in Bengali cuisine, including fish curries, sweets like rasgulla, and mustard-based dishes.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    ),
    ConversableAgent(
        name="Gujarati_Chef",
        system_message="Expert in Gujarati cuisine, including dhokla, thepla, and sweet-savory vegetarian dishes.",
        llm_config=llm_config,
        human_input_mode="NEVER",
    ),
]

# Create a Restaurant Manager Agent
restaurant_manager_agent = ConversableAgent(
    name="Restaurant_Manager",
    system_message="You assist customers with questions about different cuisines and dishes at the restaurant.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Initialize Group Chat
restaurant_chat = GroupChat(
    agents=[restaurant_manager_agent] + chefs,
    messages=[],
    max_round=max_turns,
)

# Create Group Chat Manager
restaurant_chat_manager = GroupChatManager(
    groupchat=restaurant_chat,
    llm_config=llm_config,
)

# Initiate Chat with the Restaurant Manager
chat_result = restaurant_manager_agent.initiate_chat(
    restaurant_chat_manager,
    message="What are the key ingredients in an authentic rajma chawal?",
    summary_method="reflection_with_llm",
)