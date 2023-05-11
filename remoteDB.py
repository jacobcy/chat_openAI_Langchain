import os
import json
import logging
from colorama import Fore, Style

from config import config
config()

store_file = "./backup/pinecone.json"
if not os.path.exists(store_file):
    with open(store_file, "w") as f:
        json.dump({}, f)

# 初始化 pinecone
import pinecone
pinecone.init(
    api_key = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENVIRONMENT")
)
index_name = "langchain-pinecone-hybrid-search"

# 初始化bm25_encoder
from bm25 import load_bm25
bm25_encoder = load_bm25("./bm25.json")

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(temperature=0)

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 文档工具
from documents import Docs
document = Docs()

# 初始化向量查询接口
from langchain.vectorstores import Pinecone
docsearch = Pinecone.from_existing_index(
            index_name,
            embeddings
            )

from langchain.retrievers import PineconeHybridSearchRetriever
if bm25_encoder:
    # use hybrid model
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=pinecone.Index(index_name)
        )
else:
    # use simple model
    retriever = docsearch.as_retriever()

# 创建一个构建和查询数据索引的类

class Index:
    def get_docsearch():
        return docsearch

    def get_retriever():
        return retriever

    # 查询pinecone索引
    def query(input_text):
        if bm25_encoder:
            result = retriever.get_relevant_documents(input_text)
        else:
            result = docsearch.similarity_search(
                input_text,
                include_metadata=True
                )
        result = result[:2]
        text = "\n-----------\n".join([t.page_content for t in result])
        logging.info(f"""
pinecone search for '{input_text}':
There are {len(result)} results.
The most relevant results are:
{text}
            """)
        return result

    # upload file to pinecone, and save file name
    def upload_file(file_path):
        with open(store_file, "r", encoding="utf-8") as f:
            store = json.load(f)
        if file_path in store:
            print(Fore.BLUE + f"{file_path} has been uploaded" + Style.RESET_ALL)
            return

        # add file to pinecone
        docs = document.get_docs_from(file_path)
        for t in docs:
            try:
                retriever.add_texts(t.page_content)
            except:
                logging.error(f"add_text error: \n{t.page_content[:100]}")

        # save file name
        store[file_path] = True
        print(Fore.BLUE + f"upload {file_path} to pinecone" + Style.RESET_ALL)
        with open(store_file, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=4)

    def upload_files(directory, file_type="pdf"):
        files = document.get_files_from_directory(directory, file_type)
        for file in files:
            Index.upload_file(file)

if __name__ == "__main__":
    print(Fore.GREEN + "pinecone init:" ,pinecone.whoami() + Style.RESET_ALL)

    # 初始化远程索引
    Index.upload_files("./paper","pdf")

    ## create the index
    # pinecone.create_index(
    #     index_name,
    #     dimension = 1536,  # dimensionality of dense model
    #     metric = "dotproduct",  # sparse values supported only for dotproduct
    #     pod_type = "Starer",
    #     metadata_config = {"indexed": []}  # see explaination above
    # )