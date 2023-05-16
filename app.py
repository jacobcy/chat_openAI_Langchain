
# 导入环境变量
from config import config
config()

if __name__ == "__main__":

    from chatbot import Chatbot
    chat = Chatbot()

    # agent example
    # tools = chat.get_tools("What could I do today with my kiddo?")
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

    # text = chat.query(text)
    # chat.query_google(text)
    # chat.query_local(text)
    # chat.query_remote(text)
    chat.qa_prompt(text)

    # 生成前端界面
    # import gradio as gr
    # iface = gr.Interface(fn=chat.query_local_index,
    #                      inputs=gr.inputs.Textbox(
    #                         lines=7,
    #                         label="请输入，您想从知识库中获取什么？"
    #                         ),
    #                      outputs="text",
    #                      title="AI 本地知识库ChatBot"
    #                      )
    # iface.launch(share=True)


