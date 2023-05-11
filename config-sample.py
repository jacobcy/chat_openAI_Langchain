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

# 目录配置
folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)  # 文件路径

# 日志配置
import time
day = time.strftime("%Y_%m_%d", time.localtime())
log_path = os.path.join(folder, "log", f"main_{day}.log")
print(f"Log path: {log_path}")

# 初始化日志
import logging
logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode="a",
    format="%(asctime)s - %(levelname)s: %(message)s",
    encoding="utf-8")
logging.info(f"\nStart running...")