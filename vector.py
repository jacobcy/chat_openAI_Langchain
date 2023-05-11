import logging
from config import config

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from storage import vector_store

# 创建一个本地和远程创建和查询数据索引的类

class Index:
    def __init__(self):
        from langchain.chat_models import ChatOpenAI
        self.llm = ChatOpenAI(temperature=0)
        self.max_chunk_overlap = 20
        self.chunk_size_limit = 300

        # 初始化 openai embeddings
        self.embeddings = OpenAIEmbeddings()
        self.docsearch, self.retriever = vector_store(self.embeddings)

    # 切割文本
    def split_document(self, document):
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size_limit,
            chunk_overlap=self.max_chunk_overlap
            )
        # 如果document是数组
        if isinstance(document, list):
            docs = text_splitter.split_documents(document)
        else:
            docs = text_splitter.split_text(document)

        print(f"split docs: {len(docs)}")
        document = "\n---\n".join([t.page_content for t in docs])
        logging.info(f"content:\n{document}")
        return docs

    # 从文本中获取文档
    def get_docs_from_text(self, file_path):
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file_path)
        document = loader.load()
        docs = self.split_document(document)
        logging.info(f"{file_path}:fetched {len(docs)} docs")
        return docs

    # 从pdf中获取文档
    def get_docs_from_pdf(self, file_path):
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file_path)
        document = loader.load()
        docs = self.split_document(document)
        logging.info(f"{file_path}:fetched {len(docs)} docs")
        return docs

    # 从文件中获取文档
    def get_docs_from_file(self, file_path):
        from langchain.document_loaders import UnstructuredFileLoader
        loader = UnstructuredFileLoader(file_path)
        document = loader.load()
        docs = self.split_document(document)
        logging.info(f"{file_path}:fetched {len(docs)} docs")
        return docs

    # 从目录中获取文档
    def get_docs_from_directory(self, directory_path, file_type="docx"):
        from langchain.document_loaders import DirectoryLoader
        loader = DirectoryLoader(directory_path, glob=f"**/*.{file_type}")
        documents = loader.load()
        print(f"documents: {len(documents)}")
        docs = self.split_document(documents)
        logging.info(f"{directory_path}:fetched {len(docs)} docs")
        return docs

    #  总结文档
    def summarize(self, file_path="./README.md", num=0):
        if file_path.endswith(".txt"):
            docs = self.get_docs_from_text(file_path)
        else:
            docs = self.get_docs_from_file(file_path)
        # 创建总结链
        from langchain.chains.summarize import load_summarize_chain
        chain = load_summarize_chain(
            llm = self.llm,
            chain_type="map_reduce", # or "refine"
            verbose=True)
        # 执行总结链
        if num > 0:
            docs = docs[:num]
        chain.run(docs)

    # 构建Pinecone索引
    def build_remote_index(self, directory_path, file_type="docx"):
        docs = self.get_docs_from_directory(directory_path, file_type)
        # document 计算 embedding 向量信息并存入向量数据库，用于后续匹配查询
        # Pinecone.from_texts(
        #     [t.page_content for t in docs],
        #     self.embeddings,
        #     index_name=self.index_name
        #     )

        # add_texts() 方法可以一次性添加多个文档
        self.retriever.add_texts([t.page_content for t in docs])

    # 查询pinecone索引
    def query_remote_index(self, input_text):
        result = self.retriever.get_relevant_documents(input_text)
        text = "\n---\n".join([t.page_content for t in result])
        logging.info(f"text:\n{text}")
        return result

        # 数据检索
        # result = self.docsearch.similarity_search(
        #     input_text,
        #     include_metadata=True
        #     )
        # logging.info(f"result: {result}")
        # return result

    # 数据持久化
    def local_storage(self, docs, directory="db"):
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=directory
            )
        vectordb.persist()
        vectordb = None

    # 构建本地索引
    def build_local_index(self, directory_path, file_type):
        # 循环遍历目录下的所有文件，返回文件名
        import os
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file_type:
                    if not file.endswith(file_type):
                        continue
                file_path = os.path.join(root, file)
                print(f"file_path: {file_path}")
                if file_path.endswith(".txt"):
                    docs = self.get_docs_from_text(file_path)
                elif file_path.endswith(".pdf"):
                    docs = self.get_docs_from_pdf(file_path)
                else:
                    docs = self.get_docs_from_file(file_path)
                text = "\n---\n".join([t.page_content for t in docs])
                logging.info(f"{file}:\n{text}")
                self.local_storage(docs)

    # 查询本地索引
    def query_local_index(self, input_text, directory="db"):
        # Now we can load the persisted database from disk
        vectordb = Chroma(
            embedding_function=self.embeddings,
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

        text = "\n---\n".join([t.page_content for t in result])
        logging.info(f"text:\n{text}")
        return result


if __name__ == "__main__":
    config()
    index = Index()

    # 初始化远程索引
    # index.build_remote_index("./paper","pdf")

    # 初始化本地索引
    index.build_local_index("./paper", "docx")

    # 总结测试
    # index.summarize("./paper/长效药.docx", 10)

