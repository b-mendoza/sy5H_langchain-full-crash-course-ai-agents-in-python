import requests
from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
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


class GetWeatherResponse(BaseModel):
    city: str
    temperature: float
    description: str


@tool(
    description="Return weather information for a given city",
    name_or_callable="get_weather",
    return_direct=False,
)
def get_weather(city: str) -> GetWeatherResponse:
    # TODO: use a real weather API here, this is just a placeholder
    return GetWeatherResponse(
        city=city,
        description="It's always sunny!",
        temperature=25.0,
    )


llm = ChatOpenAI(
    api_key=validated_env_vars.OPENAI_API_KEY,
    model="gpt-4.1-mini-2025-04-14",
)


agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather],
)


user_message = HumanMessage(
    content="What's the weather like in New York City?",
)


agent_response = agent.invoke(
    input={
        "messages": [
            user_message,
        ]
    }
)


class AgentResponseMessage(BaseModel):
    content: str


class AgentResponse(BaseModel):
    messages: list[AgentResponseMessage]


print("RAW agent response:", agent_response)

validated_agent_response = AgentResponse.model_validate(
    agent_response,
)

print(validated_agent_response.messages[-1].content)
