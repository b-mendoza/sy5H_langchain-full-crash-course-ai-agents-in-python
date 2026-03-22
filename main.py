from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain.messages import AnyMessage, HumanMessage
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


class AgentContext(BaseModel):
    user_id: str


class AgentResponseFormat(BaseModel):
    humidity: float
    summary: str
    temperature_in_celsius: float
    temperature_in_fahrenheit: float


class GetWeatherResponse(BaseModel):
    city: str
    description: str
    temperature: float


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


@tool(
    description="Locate the user based on the context",
    name_or_callable="locate_user",
)
def locate_user(runtime: ToolRuntime[AgentContext]) -> str:
    match runtime.context.user_id:
        case "ABC123":
            return "New York City"
        case "XYZ456":
            return "San Francisco"
        case "HJKL111":
            return "Los Angeles"
        case _:
            return "Unknown location"


model = ChatOpenAI(
    api_key=validated_env_vars.OPENAI_API_KEY,
    model="gpt-4.1-mini-2025-04-14",
    temperature=0.3,
)

checkpointer = InMemorySaver()


agent = create_agent(
    checkpointer=checkpointer,
    context_schema=AgentContext,
    model=model,
    response_format=AgentResponseFormat,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        get_weather,
        locate_user,
    ],
)


user_message = HumanMessage(
    content="What's the weather like?",
)

config = RunnableConfig(
    configurable={
        "thread_id": 1,
    }
)


agent_response = agent.invoke(
    config=config,
    context=AgentContext(
        user_id="ABC123",
    ),
    input={
        "messages": [
            user_message,
        ]
    },
)


class AgentResponse(BaseModel):
    messages: list[AnyMessage]


print(
    "RAW agent response:",
    agent_response,
)

validated_agent_response = AgentResponse.model_validate(
    agent_response,
)

last_message_content = validated_agent_response.messages[-1].content

if isinstance(last_message_content, str):
    print(last_message_content)
