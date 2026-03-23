from typing import Literal

from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, SecretStr

ENV_VARS_FILE_PATH = ".env"


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


class AgentContext(BaseModel):
    user_role: Literal["expert", "beginner", "child"]


@dynamic_prompt
def user_role_prompt(request: ModelRequest[AgentContext]) -> str:
    base_prompt = "You are a helpful and very concise assistant."

    match request.runtime.context.user_role:
        case "expert":
            return f"""{base_prompt} Provide detailed and technical
            explanations."""
        case "beginner":
            return f"""{base_prompt} Provide simple and easy-to-understand
            explanations."""
        case "child":
            return f"""{base_prompt} Provide explanations as if you were
            literally talking to a 5-year-old."""


model = ChatOpenAI(
    api_key=validated_env_vars.OPENAI_API_KEY,
    model="gpt-4.1-mini-2025-04-14",
)

agent = create_agent(
    model=model,
    middleware=[
        user_role_prompt,
    ],
    context_schema=AgentContext,
)

agent_response = agent.invoke(
    input={
        "messages": [
            HumanMessage(
                content="Explain PCA.",
            )
        ]
    },
    context=AgentContext(
        user_role="child",
    ),
)

print(agent_response)
