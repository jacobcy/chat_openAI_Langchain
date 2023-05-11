import os
import json
import logging
from colorama import Fore, Style

from config import config
config()

from documents import Docs
document = Docs()

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

store_file = "./backup/chroma.json"
if not os.path.exists(store_file):
    with open(store_file, "w") as f:
        json.dump({}, f)

# 使用Chroma本地处理数据
class LocalDB:
    # 查询本地索引
    def query(input_text, directory="db"):
        # Now we can load the persisted database from disk
        vectordb = Chroma(
            embedding_function=embeddings,
            persist_directory=directory
            )

        # search for similar documents
        # result = vectordb.similarity_search(input_text)
        # vectordb = None

        # search for retrieval documents
        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
            )
        result = retriever.get_relevant_documents(input_text)

        text = "\n-----------\n".join([t.page_content for t in result])
        logging.info(f"chroma search for '{input_text}':\n{text}")
        return result

    # 数据持久化
    def save(docs, directory="db"):
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=directory
            )
        vectordb.persist()
        vectordb = None

    # upload file, and save file name to store.json
    def upload_file(file_path):
        with open(store_file, "r", encoding="utf-8") as f:
            store = json.load(f)
        if file_path in store:
            print(Fore.BLUE + f"{file_path} has been uploaded" + Style.RESET_ALL)
            return

        docs = document.get_docs_from(file_path)
        text = "\n-----------\n".join([t.page_content for t in docs])
        logging.info(f"{file_path}: \n{text}")
        try:
            LocalDB.save(docs)
        except:
            logging.error(f"save {file_path} to chroma failed")
            return
        # save file name
        store[file_path] = True
        print(Fore.GREEN + f"upload {file_path} to pinecone" + Style.RESET_ALL)
        with open(store_file, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=4)

    def upload_files(directory, file_type="pdf"):
        files = document.get_files_from_directory(directory, file_type)
        for file in files:
            LocalDB.upload_file(file)

if __name__ == "__main__":
    # 初始化本地索引
    LocalDB.upload_files("./paper", "docx")
