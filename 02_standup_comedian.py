import os

from dotenv import load_dotenv

load_dotenv()
from autogen import ConversableAgent

azure_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

llm_config={"config_list": [
        {"model": azure_model_name, "api_key": api_key, 'base_url': azure_endpoint, "api_type": "azure",
         "api_version": "2024-02-01"}]}

cathy = ConversableAgent(
    "cathy",
    system_message="Your name is Cathy and you are a part of a duo of comedians.",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
    max_consecutive_auto_reply=2,

)

joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config=llm_config,
    human_input_mode="NEVER",  # Never ask for human input.
    max_consecutive_auto_reply=2,
    is_termination_msg=lambda msg:"good bye" in msg['content'].lower()
)

result = joe.initiate_chat(cathy, message="Cathy, tell me a joke and then say words good bye")
print(result)
