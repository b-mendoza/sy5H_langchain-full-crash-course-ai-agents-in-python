from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.runnables import RunnableConfig
from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, SecretStr

ENV_VARS_FILE_PATH = ".env"

SYSTEM_PROMPT = """You are a helpful weather assistant who always cracks jokes 
and is humorous while remaining helpful."""


class EnvVars(BaseModel):
    ANTHROPIC_API_KEY: SecretStr
    GEMINI_API_KEY: SecretStr
    MISTRAL_API_KEY: SecretStr
    OPENAI_API_KEY: SecretStr
    WEATHER_API_KEY: SecretStr


validated_env_vars = EnvVars.model_validate(
    dotenv_values(
        dotenv_path=ENV_VARS_FILE_PATH,
    )
)

model = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    api_key=validated_env_vars.OPENAI_API_KEY,
)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe the content of this image",
        },
        {
            "type": "image",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/640px-PNG_transparency_demonstration_1.png",
        },
    ]
)

response = model.invoke(
    input=message,
)
