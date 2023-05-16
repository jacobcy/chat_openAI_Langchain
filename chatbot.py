
from config import config
config()

import logging
from colorama import Fore, Style

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from langchain.tools import BaseTool
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import ChatVectorDBChain, ConversationChain, ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from langchain.schema import Document
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate
)

from remote_db import Index
from local_db import LocalDB
from documents import Docs

# 创建一个类：通过openai api实现chatbot功能
class Chatbot:

    # 初始化
    max_input_size = 512
    num_outputs = 512
    llm = ChatOpenAI(
        temperature=0,
        max_tokens=num_outputs,
        streaming=True,
        verbose=True
        )
    # 初始化 MessageHistory 对象
    history = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = Index.retriever

    def save_history(func):
        def wrapper(question):
            answer = func(question)
            print(Fore.CYAN + f"answer: {answer}" + Style.RESET_ALL)
            # 给 MessageHistory 对象添加对话内容，需要添加字符串
            Chatbot.history.add_user_message(question)
            Chatbot.history.add_ai_message(answer)
            # chatgpt.history的内容存放在chatgpt.history.messages中
            logging.info(f"Chat history: \n{Chatbot.history}")
            return answer
        return wrapper
    
    def qa(question: str) -> str:
        """
        description: 问答
        :param question: str
        return: str
        """
        messages = [
            SystemMessage(content="You are a helpful GPT-5 assistant, always think step by step."),
            HumanMessage(content=question)
        ]
        message = Chatbot.llm(messages)
        answer = message.content
        return answer

    def translate(input_text: str) -> str:
        """
        description: 翻译
        :param input_text: str
        return: str
        """
        target = f"Translate English to Chinese: {input_text}"
        output_text = Chatbot.qa(target)
        print(Fore.MAGENTA + output_text + Style.RESET_ALL)
        return output_text

    def summarize(docs: list) -> list:
        """
        description: 总结文档
        :param docs: list
        return: list
        """
        chain = load_summarize_chain(
            llm = Chatbot.llm,
            chain_type="map_reduce", # or "refine"
            verbose=True)
        brief = chain.run(docs)
        print(Fore.MAGENTA + brief + Style.RESET_ALL)
        return Document(brief)

    def run_qa_chain(context :list, question: str) -> str:
        """
        description: 根据上下文，回答问题
        :param context: list[Document]
        :param question: str
        return: str
        """
        chain = load_qa_chain(Chatbot.llm, chain_type="stuff", verbose=True)
        answer = chain.run(input_documents=context, question=question)
        return answer
    
    @save_history
    def qa_remote(question: str) -> str:
        """
        description: 从远程知识库获取答案
        :param question: str
        return: str
        """
        context = Index.query(question)
        answer = Chatbot.run_qa_chain(context, question)
        return answer

    @save_history
    def qa_local(question: str) -> str:
        """
        description: 从本地知识库获取答案
        :param question: str
        return: str
        """
        context = LocalDB.query(question)
        answer = Chatbot.run_qa_chain(context, question)
        return answer

    @save_history
    def qa_google(question: str) -> str:
        """
        description: 查询谷歌搜索
        :param question: str
        return: str
        """
        tools = load_tools(["serpapi"])
        # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
        agent = initialize_agent(
            tools=tools,
            llm=Chatbot.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        answer = agent.run(question)
        return answer

    def predict(input_text: str) -> str:
        """
        description: 根据模板生成prompt,然后回答问题,保持对话记忆
        :param input_text: str
        return: str
        """
        template = """
        The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {history}
        Friend: {input}
        AI:"""
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], 
            template=template
        )
        conversation = ConversationChain(
            prompt=PROMPT,
            llm=Chatbot.llm,
            verbose=True,
            memory=ConversationBufferMemory(human_prefix="Friend")
        )
        output_text = conversation.predict(input=input_text)
        return output_text

    # @save_history
    def qa_paper(brief: str) -> str:
        """
        description: 构造prompt进行提问,包含远程知识库
        :param brief: str
        return: str
        """
        system_template = """
            Based on the following context, suggest modifications and provide an improved version.
            If you don't know the answer, say you don't, don't try to make it up.
            -----------
            {context}
            -----------
            {chat_history}
            """
        human_template = """
            Pretend as a proffesor, optimize student's paper:
            -----------
            {question}
            """

        # 构建初始 messages 列表
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
            ]
        # 初始化 prompt 对象
        prompt = ChatPromptTemplate.from_messages(messages)

        qa = ConversationalRetrievalChain.from_llm(
            Chatbot.llm,
            Chatbot.retriever,
            qa_prompt=prompt
            )

        result = qa({'question': brief
            # , 'chat_history': chatgpt.history.messages
            })
        answer = result['answer']
        return answer

    @save_history
    def qa_prompt(question: str) -> str:
        """
        description: 构造prompt进行提问,包含远程知识库
        :param question: str
        return: str
        """
        question_generator = LLMChain(llm=Chatbot.llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_with_sources_chain(Chatbot.llm, chain_type="map_reduce")

        chain = ConversationalRetrievalChain(
            retriever=Chatbot.retriever,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
        )

        qa = ConversationalRetrievalChain.from_llm(
            Chatbot.llm,
            Chatbot.retriever,
            memory=Chatbot.memory
            )

        result = qa({'question': question,
            'chat_history': Chatbot.history.messages
            })
        answer = result['answer']
        return answer


if __name__ == "__main__":

    chat = Chatbot()

    text = "What is the Apretude?"
    # chatbot.predict(text)
    # chatbot.translate(text)

    # chatbot.qa_google(text)
    # chatbot.qa_local(text)
    # text = chatbot.qa_remote(text)
    # text = chatbot.translate(text)


    # 命令行对话
    # while True:
    #     question = input("问题(x结束):\n")
    #     if question == "x":
    #         break
    #     answer = chatbot.query(question)
    #     print((f"回答:\n{answer}"))

    # 搜索
    # chatbot.query_google("What is the best programming language?")

    paper = "./docs/Manuscript-LP98-MS-xuejing.docx"
    document = Docs.get_docs_from(paper)
    for doc in document[:2]:
        doc = chat.summarize(doc)
        print(Fore.CYAN + f"摘要: \n{doc.page_content}" + Style.RESET_ALL)
        chat.qa_prompt(doc.page_content)