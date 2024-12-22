from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

apy_key = os.getenv("OPENAI_API_KEY")

# Crear una instancia de ChatOpenAI con el modelo GPT-4o mini
chat_model = ChatOpenAI(openai_api_key=apy_key, model="gpt-4o-mini")

template = (
    "You are a helpful assistant that translate {input_language} to {output_language}."
)
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", human_template)]
)

messages = chat_prompt.format_messages(
    input_language="English", output_language="Spanish", text="I'm learning langchain"
)

result = chat_model.invoke(messages)

print(result.content)
