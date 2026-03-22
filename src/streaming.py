from dotenv import dotenv_values
from langchain_mistralai.chat_models import ChatMistralAI
from pydantic import BaseModel, SecretStr

ENV_VARS_FILE_PATH = ".env"

SYSTEM_PROMPT = """You are a helpful coding assistant who can help with Python questions and issues."""


class EnvVars(BaseModel):
    ANTHROPIC_API_KEY: SecretStr
    GEMINI_API_KEY: SecretStr
    MISTRAL_API_KEY: SecretStr
    OPENAI_API_KEY: SecretStr


validated_env_vars = EnvVars.model_validate(
    dotenv_values(
        dotenv_path=ENV_VARS_FILE_PATH,
    )
)


model = ChatMistralAI(
    api_key=validated_env_vars.MISTRAL_API_KEY,
    model_name="mistral-medium-2508",
    temperature=0.3,
)

model_response = model.stream(
    input="What is Python?",
)

for chunk in model_response:
    model_completion_content = chunk.content

    if isinstance(model_completion_content, str):
        print(
            model_completion_content,
            end="",
        )
