import logging

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from langchain.tools import BaseTool
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

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
from vector import Index

# 创建一个类：通过openai api实现chatbot功能
class chatgpt:
    def __init__(self):
        self.index = Index()
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

    def save_history(func):
        def wrapper(self, question):
            answer = func(self, question)
            print(f"answer: {answer}")
            # 给 MessageHistory 对象添加对话内容，需要添加字符串
            self.history.add_user_message(question)
            self.history.add_ai_message(answer)
            # self.history的内容存放在self.history.messages中
            logging.info(f"Chat history: \n{self.history.messages}")
        return wrapper

    # 包含历史记录的对话
    @save_history
    def query(self, question):
        messages = [
            SystemMessage(content="You are a helpful GPT-5 assistant, always think step by step."),
            HumanMessage(content=question)
        ]
        message = self.llm(messages)
        answer = message.content
        return answer

    # 构造prompt进行提问
    @save_history
    def query_prompt(self, question):
        system_template = """
            Use the following context to answer the user's question.
            If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
            -----------
            {context}
            -----------
            {chat_history}
            """
        # 构建初始 messages 列表
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
            ]
        # 初始化 prompt 对象
        prompt = ChatPromptTemplate.from_messages(messages)

        qa = ConversationalRetrievalChain.from_llm(
            self.llm,
            self.index.retriever,
            qa_prompt=prompt
            )
        result = qa({'question': question,
            'chat_history': self.history.messages
            })
        answer = result['answer']
        print(f"answer: {answer}")
        return answer

    # 查询谷歌搜索,agent返回答案为字符串
    @save_history
    def query_google(self, question):
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
    @save_history
    def query_remote(self, question):
        context = self.index.query_remote_index(question)
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
        print("answer: ", str(answer))
        return answer

    # 从本地知识库获取答案
    @save_history
    def query_local(self, question):
        context = self.index.query_local_index(question)
        from langchain.chains.question_answering import load_qa_chain
        chain = load_qa_chain(
            self.llm,
            chain_type="stuff",
            verbose=True
            )
        answer = chain.run(input_documents=context, question=question)
        return answer

    # 构造prompt进行提问
    def gen_prompt(self, template, input_text):
        prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )
        output_text = prompt.format(question=input_text)
        logging.info(f"prompt: \n{output_text}")
        return output_text

    # 翻译
    def translate(self, input_text):
        target = "Translate English to Chinese:{question}"
        output_text = self.gen_prompt(target, input_text)
        return output_text

if __name__ == "__main__":

    from config import config
    config()
    chatbot = chatgpt()
    # 命令行对话
    while True:
        question = input("问题(x结束):\n")
        if question == "x":
            break
        answer = chatbot.query(question)
        print((f"回答:\n{answer}"))

    # 搜索
    # chatbot.query_google("What is the best programming language?")