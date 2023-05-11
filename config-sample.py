# 设置环境变量
import os
def config():
    # 初始化 openai
    os.environ["OPENAI_API_KEY"] = ""

    # 初始化 serpapi
    os.environ["SERPAPI_API_KEY"] = ""

    # 初始化 pinecone
    os.environ["PINECONE_API_KEY"] = ""
    os.environ["PINECONE_ENVIRONMENT"] = ""