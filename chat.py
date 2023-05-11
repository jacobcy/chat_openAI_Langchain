
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

from remoteDB import Index
from localDB import LocalDB

# 创建一个类：通过openai api实现chatbot功能
class chatgpt:
    def __init__(self):

        # 初始化
        self.max_input_size = 512
        self.num_outputs = 512

        self.llm = ChatOpenAI(
            temperature=0,
            max_tokens=self.num_outputs,
            streaming=True,
            verbose=True
            )
        # 初始化 MessageHistory 对象
        self.history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.retriever = Index.get_retriever()

    def save_history(func):
        def wrapper(self, question):
            answer = func(self, question)
            print(Fore.CYAN + Style.DIM)
            print(f"answer: {answer}")
            print(Style.RESET_ALL)
            # 给 MessageHistory 对象添加对话内容，需要添加字符串
            self.history.add_user_message(question)
            self.history.add_ai_message(answer)
            # self.history的内容存放在self.history.messages中
            logging.info(f"Chat history: \n{self.history}")
            return answer
        return wrapper

    # 包含历史记录的对话
    # @save_history
    def qa(self, question):
        messages = [
            SystemMessage(content="You are a helpful GPT-5 assistant, always think step by step."),
            HumanMessage(content=question)
        ]
        message = self.llm(messages)
        answer = message.content
        return answer

    # 翻译
    def translate(self, input_text):
        target = f"Translate English to Chinese:{input_text}"
        output_text = self.qa(target)
        print(Fore.MAGENTA + Style.NORMAL)
        print(output_text)
        print(Style.RESET_ALL)

    #  总结文档
    def summarize(self, docs):
        # 如果docs是字符串
        if isinstance(docs, str):
            docs = [Document(page_content=docs)]
        chain = load_summarize_chain(
            llm = self.llm,
            chain_type="map_reduce", # or "refine"
            verbose=True)
        brief = chain.run(docs)
        print(Fore.MAGENTA + Style.NORMAL)
        print(brief)
        print(Style.RESET_ALL)
        return [Document(page_content=brief)]

    # 构造prompt进行提问,包含远程知识库
    @save_history
    def qa_paper(self, brief):
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
            self.llm,
            self.retriever,
            qa_prompt=prompt
            )

        result = qa({'question': brief
            # , 'chat_history': self.history.messages
            })
        answer = result['answer']
        return answer

    @save_history
    def qa_prompt(self, question):
        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_with_sources_chain(self.llm, chain_type="map_reduce")

        chain = ConversationalRetrievalChain(
            retriever=self.retriever,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
        )

        qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.retriever,
            memory=self.memory
            )

        result = qa({'question': question,
            'chat_history': self.history.messages
            })
        answer = result['answer']
        return answer

    # 查询谷歌搜索,agent返回答案为字符串
    @save_history
    def qa_google(self, question):
        tools = load_tools(["serpapi"])
        # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory = ConversationBufferMemory() # 这里并没有存储到self.history中
            # return_intermediate_steps=True
        )
        answer = agent.run(question)
        return answer

    # 从远程知识库获取答案
    def qa_remote(self, question):
        context = Index.query(question)
        # context = self.summarize(context)
        logging.info(f"Context for {question}: \n{context}")
        from langchain.chains.question_answering import load_qa_chain
        chain = load_qa_chain(
            self.llm,
            chain_type="stuff",
            verbose=True
            )
        answer = chain.run(
            input_documents=context,
            question=question
            )
        print(Fore.LIGHTBLUE_EX + "answer: ", str(answer) + Style.RESET_ALL)
        return answer

    # 从本地知识库获取答案
    @save_history
    def qa_local(self, question):
        context = LocalDB.query(question)
        from langchain.chains.question_answering import load_qa_chain
        chain = load_qa_chain(
            self.llm,
            chain_type="stuff",
            verbose=True
            )
        answer = chain.run(input_documents=context, question=question)
        return answer

    @save_history
    def predict(self, input_text):
        template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

        Current conversation:
        {history}
        Friend: {input}
        AI:"""
        PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=template
        )
        conversation = ConversationChain(
            prompt=PROMPT,
            llm=self.llm,
            verbose=True,
            memory=ConversationBufferMemory(human_prefix="Friend")
        )
        output_text = conversation.predict(input=input_text)
        return output_text

if __name__ == "__main__":

    chatbot = chatgpt()

    text = "What is the Apretude?"
    # chatbot.predict(text)
    # chatbot.translate(text)

    # chatbot.qa_google(text)
    # chatbot.qa_local(text)
    text = chatbot.qa_remote(text)
    text = chatbot.translate(text)


    # 命令行对话
    # while True:
    #     question = input("问题(x结束):\n")
    #     if question == "x":
    #         break
    #     answer = chatbot.query(question)
    #     print((f"回答:\n{answer}"))

    # 搜索
    # chatbot.query_google("What is the best programming language?")

    # paper = "./docs/Manuscript-LP98-MS-xuejing.docx"
    # from documents import Docs
    # docs = Docs()
    # document = docs.get_docs_from(paper)
    # for doc in document[:2]:
    #     brief = chatbot.summarize(doc)
    #     print(Fore.CYAN + f"摘要: \n{brief}" + Style.RESET_ALL)
    #     chatbot.qa_paper(brief)