# 导入环境变量
from config import config
config()

# 目录配置
import os
folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)  # 文件路径

# 日志配置
import time
import logging
times = int(time.time())
log_path = os.path.join(folder, "log", f"main_{times}.log")
print(f"Log path: {log_path}")

logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s: %(message)s",
    encoding="utf-8")
logging.info(f"Start running...")

from chat import chatgpt

# import gradio as gr

if __name__ == "__main__":

    chatbot = chatgpt()

    # agent example
    # tools = chatbot.get_tools("What could I do today with my kiddo?")
    # tool_names = [t.name for t in tools]
    # tool_text = "\n".join(tool_names)
    # logging.info(f"tool_names: \n{tool_text}")

    # 命令行提问
    file = r"docs\Manuscript-LP98-MS-xuejing.docx"

    text = """
    when I run the code:
    ---
    docsearch = Pinecone.from_existing_index(
        index_name,
        embeddings
        )
    docsearch.similarity_search(input_text)
    ---
    error shows:
    ---
    text = metadata.pop(self._text_key)
    KeyError: text
    ---
    How to fix it?
    """
    text = "Tell me about the GS-6207?"

    # text = chatbot.query(text)
    # chatbot.query_google(text)
    # chatbot.query_local(text)
    # chatbot.query_remote(text)
    chatbot.query_prompt(text)

    # 生成前端界面
    # iface = gr.Interface(fn=chatbot.query_local_index,
    #                      inputs=gr.inputs.Textbox(
    #                         lines=7,
    #                         label="请输入，您想从知识库中获取什么？"
    #                         ),
    #                      outputs="text",
    #                      title="AI 本地知识库ChatBot"
    #                      )
    # iface.launch(share=True)


