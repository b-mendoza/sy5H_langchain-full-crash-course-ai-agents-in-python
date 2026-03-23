from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import BaseModel, SecretStr

ENV_VARS_FILE_PATH = ".env"

SYSTEM_PROMPT = """You are a helpful assistant who returns relevant information
from a knowledge base when asked about fruits or computers. You should use the
kb_search tool to find relevant information or use it multiple times to aggregate
information from multiple documents. Always concisely return the relevant information
and avoid returning irrelevant information."""


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

embeddings = OpenAIEmbeddings(
    api_key=validated_env_vars.OPENAI_API_KEY,
    model="text-embedding-3-large",
)

texts = [
    "Apple makes very good computers.",
    "I believe Apple is innovative.",
    "I love eating apples.",
    "I'm a fan of MacBooks.",
    "I enjoy oranges.",
    "I love Lenovo ThinkPads.",
    "I think pears taste very good.",
]

vector_store = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
    }
)

retriever_tool = create_retriever_tool(
    description="""Search the small product / fruit knowledge base for relevant
    information.""",
    name="kb_search",
    retriever=retriever,
)

model = ChatOpenAI(
    api_key=validated_env_vars.OPENAI_API_KEY,
    model="gpt-4.1-mini-2025-04-14",
)


class AgentContext(BaseModel):
    user_id: str


class AgentResponseFormat(BaseModel):
    relevant_information: str


agent = create_agent(
    context_schema=AgentContext,
    model=model,
    response_format=AgentResponseFormat,
    system_prompt=SYSTEM_PROMPT,
    tools=[retriever_tool],
)

agent_response = agent.invoke(
    context=AgentContext(
        user_id="ABC123",
    ),
    input={
        "messages": [
            HumanMessage(
                content="What three fruits and computer brands does the person like?"
            )
        ]
    },
    version="v2",
)

print(agent_response)
