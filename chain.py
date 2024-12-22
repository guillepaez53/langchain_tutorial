from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

apy_key = os.getenv("OPENAI_API_KEY")


class CommaSeparatedOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


# Crear una instancia de ChatOpenAI con el modelo GPT-4o mini
chat_model = ChatOpenAI(openai_api_key=apy_key, model="gpt-4o-mini")

template = """You are a helpful assistant who generates comma separated lists.
            A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
            ONLY return a comma separated list, and nothing else.
            """

human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", human_template)]
)

chain = chat_prompt | chat_model | CommaSeparatedOutputParser()

result = chain.invoke({"text: colors"})

print(result)
