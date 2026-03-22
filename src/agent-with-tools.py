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
    # TODO: replace with a real weather API, this is hardcoded for now
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


class AgentRuntimeConfig(BaseModel):
    thread_id: int


config = RunnableConfig(
    configurable=AgentRuntimeConfig(
        thread_id=1,
    ).model_dump(),
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
    version="v2",
)

structured_response = agent_response.value.get("structured_response")

if structured_response is not None:
    print(structured_response.summary)
    print(structured_response.temperature_in_celsius)
    print(
        "RAW structured_response:",
        structured_response,
    )

# Ask a follow-up using a different thread_id so the agent has no memory
# of the previous conversation. We want to see how it handles a question
# without any prior context.

follow_up_message = HumanMessage(
    # Contradicting the agent (e.g. "you said cold but it's sunny") breaks
    # the response_format and crashes the app.
    content="""You told me the weather would be very cold and rainy, but I see
    it's sunny and warm outside. Why?"""
    # A follow-up that doesn't contradict works fine though.
    # content="Is this weather good for a picnic?",
)


new_config = RunnableConfig(
    configurable=AgentRuntimeConfig(
        thread_id=2,
    ).model_dump(),
)

agent_response = agent.invoke(
    config=new_config,
    context=AgentContext(
        user_id="ABC123",
    ),
    input={
        "messages": [
            follow_up_message,
        ]
    },
    version="v2",
)


structured_response = agent_response.value.get("structured_response")

if structured_response is not None:
    print(structured_response.summary)
    print(structured_response.temperature_in_celsius)
    print(
        "RAW structured_response:",
        structured_response,
    )
