from base64 import b64encode

from dotenv import dotenv_values
from langchain.messages import HumanMessage, ImageContentBlock
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

model = ChatOpenAI(
    model="gpt-4.1-mini-2025-04-14",
    api_key=validated_env_vars.OPENAI_API_KEY,
)

base64_image_message: ImageContentBlock

# We use a context manager to ensure the file is properly closed after reading
with open("animal_facts-e1396431549968.jpg", "rb") as image_file:
    base64_image_message = ImageContentBlock(
        {
            "type": "image",
            "base64": b64encode(image_file.read()).decode(),
            "mime_type": "image/jpeg",
        },
    )


message = HumanMessage(
    content_blocks=[
        {
            "type": "text",
            "text": "Describe the content of this image",
        },
        # Using a URL to an image is supported
        # {
        #     "type": "image",
        #     "url": "https://africanoverlandtours.com/wp-content/uploads/2025/04/animal_facts-e1396431549968.jpg",
        # },
        # Using a base64 encoded image is also supported
        base64_image_message,
    ]
)

model_completion = model.invoke(
    input=[message],
)

completion_content = model_completion.content

if isinstance(completion_content, str):
    print(completion_content)
