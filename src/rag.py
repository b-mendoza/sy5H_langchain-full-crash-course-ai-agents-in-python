from dotenv import dotenv_values
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
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

# print(
#     vector_store.similarity_search(
#         k=7,
#         query="Apples are my favority food.",
#     )
# )

print(
    vector_store.similarity_search(
        k=7,
        query="Linux is a great operating system.",
    )
)
