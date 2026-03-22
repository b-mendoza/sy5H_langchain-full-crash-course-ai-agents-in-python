import requests
from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain.tools import tool
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


@tool(
    description="Return weather information for a given city",
    name_or_callable="get_weather",
    return_direct=False,
)
def get_weather(city: str) -> None:
    response = requests.get(
        f"https://wttr.in/{city}?format=j1",
    )

    # TODO: use Pydantic to validate the response
    return response.json()


llm = ChatOpenAI(
    api_key=validated_env_vars.OPENAI_API_KEY,
    model="gpt-4.1-mini-2025-04-14",
)


agent = create_agent(
    model=llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather],
)

agent_response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in New York City?",
            }
        ]
    }
)

print(agent_response)
print(agent_response.messages[-1].content)
