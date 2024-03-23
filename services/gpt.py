# chatbot_singleton.py
import os
from .json_loader import JSONLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from .constants import qa_system_prompt, contextualize_q_system_prompt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


class ChatbotSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()

        return cls._instance

    def initialize(self):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

        try:
            self.embedding_function = OpenAIEmbeddings()

            loader = JSONLoader(file_path="../data.json", text_content=False)
            documents = loader.load()
            self.docs = self.split_docs(documents)

            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            self.db = Chroma.from_documents(documents=self.docs, embedding=self.embedding_function)
            self.retriever = self.db.as_retriever()
            self.retriever_from_llm = MultiQueryRetriever.from_llm(retriever=self.retriever, llm=self.llm)

            # Chatbot pipeline setup
            self.qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", qa_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ]
            )

            self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                ]
            )

            self.contextualize_q_chain = self.contextualize_q_prompt | self.llm | StrOutputParser()

            self.rag_chain = (
                RunnablePassthrough.assign(
                    context=self.contextualized_question | self.retriever_from_llm | self.format_docs
                )
                | self.qa_prompt
                | self.llm
            )

        except Exception as e:
            self.error = f"The OpenAI API key does not seem to be working right now."

    @staticmethod
    def split_docs(documents, chunk_size=1000, chunk_overlap=100):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(documents)
        return docs

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return ChatbotSingleton()._instance.contextualize_q_chain
        else:
            return input["question"]
