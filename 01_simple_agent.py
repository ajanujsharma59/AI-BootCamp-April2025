import os
from dotenv import load_dotenv

load_dotenv()
from autogen import ConversableAgent

azure_model_name = os.getenv("AZURE_OPENAI_MODEL_NAME")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

agent = ConversableAgent(
    "chatbot",
    llm_config={"config_list": [
        {"model": azure_model_name, "api_key": api_key, 'base_url': azure_endpoint, "api_type": "azure",
         "api_version": "2024-02-01"}]},
    code_execution_config=False,  # Turn off code execution, by default it is off.
    function_map=None,  # No registered functions, by default it is None.
    human_input_mode="NEVER",  # Never ask for human input.
)

reply = agent.generate_reply(messages=[{"content": "Tell me a joke.", "role": "user"}])
print(reply)
