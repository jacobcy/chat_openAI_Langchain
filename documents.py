import os
import logging
from colorama import Fore, Style

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader

# 从文件中获取数据，并将数据切片
class Docs:
    max_chunk_overlap = 0
    chunk_size_limit = 200

    def split_document(loader: object) -> list:
        """
        description: 切割document对象,返回Dcoment
        :param loader: object
        return: List[Document]
        """
        # 加载文档
        document = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=Docs.chunk_size_limit,
            chunk_overlap=Docs.max_chunk_overlap
            )
        docs = text_splitter.split_documents(document)
        document = "\n-----------\n".join([t.page_content for t in docs[:3]])
        logging.info(f"split document to content, top3 likes: \n{document}")
        return docs
    
    def split_text(text: str) -> list:
        """
        description: 切割text,返回Dcoment
        :param text: str
        return: List[Document]
        """
        text_splitter = CharacterTextSplitter(
            chunk_size=Docs.chunk_size_limit,
            chunk_overlap=Docs.max_chunk_overlap
            )
        docs = text_splitter.split_text(document)
        document = "\n-----------\n".join([t.page_content for t in docs[:3]])
        logging.info(f"split text to content, top3 likes: \n{document}")
        return docs

    # 定义日志输出格式的装饰器
    def log(func):
        def wrapper(file_path):
            docs = func(file_path)
            print(Fore.GREEN + f"{file_path}: fetched {len(docs)} docs\n" + Style.RESET_ALL)
            logging.info(f"{file_path}: fetched {len(docs)} docs")
            return docs
        return wrapper

    @log
    def get_docs_from_text(file_path: str) -> list:
        """
        description: 从文本中获取文档
        :param file_path: str
        return: List[Document]
        """
        loader = TextLoader(file_path)
        docs = Docs.split_document(loader)
        return docs

    @log
    def get_docs_from_pdf(file_path: str) -> list:
        """
        description: 从pdf中获取文档
        :param file_path: str
        return: List[Document]
        """
        loader = PyPDFLoader(file_path)
        docs = Docs.split_document(loader)
        return docs

    @log
    def get_docs_from_file(file_path: str) -> list:
        """
        description: 从其他类型文件中获取文档
        :param file_path: str
        return: List[Document]
        """
        loader = UnstructuredFileLoader(file_path)
        docs = Docs.split_document(loader)
        return docs

    def get_docs_from(file_path: str) -> list:
        """
        description: 统一接口，从文件中获取文档
        :param file_path: str
        return: List[Document]
        """
        if file_path.endswith(".txt"):
            docs = Docs.get_docs_from_text(file_path)
        elif file_path.endswith(".pdf"):
            docs = Docs.get_docs_from_pdf(file_path)
        else:
            docs = Docs.get_docs_from_file(file_path)
        return docs

    def get_files_from_directory(directory_path: str, file_type: str="pdf") -> list:
        """
        description: 循环遍历目录下的所有文件，返回文件名
        :param directory_path: str
        :param file_type: str
        return: List[file_path]
        """
        files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file_type and not file.endswith(file_type):
                    continue
                file_path = os.path.join(root, file)
                logging.warning(f"Get file_path: {file_path}")
                return file_path

    def get_docs_from_directory(directory_path: str, file_type: str="pdf") -> list:
        """
        description: 从目录中获取文档
        :param directory_path: str
        :param file_type: str
        return: List[Document]
        """
        files = Docs.get_files_from_directory(directory_path, file_type)
        for file in files:
            docs = Docs.get_docs_from(file)
            yield docs

    def save_docs_to_file(docs: list, file_path: str):
        """
        description: 将文档保存到文件中
        :param docs: list
        :param file_path: str
        return: None
        """
        document = "\n-----------\n".join([t.page_content for t in docs])
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(document)

    def save_text_to_file(text: str, file_path: str):
        """
        description: 将文本保存到文件中
        :param text: str
        :param file_path: str
        return: None
        """
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    def convert_file(file_path: str):
        """
        description: 将文件转换为文档
        :param file_path: str
        return: None
        """
        docs = Docs.get_docs_from(file_path)
        Docs.save_docs_to_file(docs, file_path+".txt")
        

if __name__ == "__main__":
    docs = Docs.convert_file('paper\综述-2020-Nanosystems Applied to HIV Infection Prevention and Treatments.pdf')