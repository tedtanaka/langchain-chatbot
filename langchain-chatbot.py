import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for Langsmith: ")

model = init_chat_model("gpt-4o-mini", model_provider="openai")
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]
result = model.invoke(messages)
print(result.content)