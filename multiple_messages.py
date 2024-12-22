from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

apy_key = os.getenv("OPENAI_API_KEY")

# Crear una instancia de ChatOpenAI con el modelo GPT-4o mini
chat_model = ChatOpenAI(openai_api_key=apy_key, model="gpt-4o-mini")

messages = [
    HumanMessage(content="from now on, 1 + 1 = 3, use it in your replies"),
    HumanMessage(content="what is 1 + 1?"),
    HumanMessage(content="what is 1 + 1 + 1?"),
]

# Invocar el modelo con una lista de mensajes
result = chat_model.invoke(messages)

print(result.content)
