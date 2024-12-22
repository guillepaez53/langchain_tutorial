from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

apy_key = os.getenv("OPENAI_API_KEY")


class AnswerOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        """Parse the output of an LLM call."""
        return text.strip().split("answer =")


# Crear una instancia de ChatOpenAI con el modelo GPT-4o mini
chat_model = ChatOpenAI(openai_api_key=apy_key, model="gpt-4o-mini")

template = """You are a helpful assistant that solves math problems and shows the steps to solve them.
            Output each step in human readable format, without LaTeX symbols or any complex formatting. Write in plain text.
            Then return the answer in the following format: answer = <answer here>.
            Make sure to output answer in all lowercases and to have exactly one space and one equal sign following it.
            For example: 'answer = 1.5'
            """

human_template = "{problem}"

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", template), ("human", human_template)]
)

messages = chat_prompt.format_messages(problem="2x^2 -5x + 3 = 0")

result = chat_model.invoke(messages)

parsed = AnswerOutputParser().parse(result.content)

steps, answer = parsed

print()
print(answer)
print()
print()
print(steps)
print()
print()
print(result.content)
