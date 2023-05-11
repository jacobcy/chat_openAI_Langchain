import os
import logging
from colorama import Fore, Style

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredFileLoader

# 从文件中获取数据，并将数据切片
class Docs:
    def __init__(self):
        self.max_chunk_overlap = 0
        self.chunk_size_limit = 200

    # 切割文本
    def split_document(self, loader):
        document = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size_limit,
            chunk_overlap=self.max_chunk_overlap
            )
        # 如果document是数组
        if isinstance(document, list):
            docs = text_splitter.split_documents(document)
        else:
            docs = text_splitter.split_text(document)
        document = "\n-----------\n".join([t.page_content for t in docs])
        logging.info(f"split document to content: \n{document}")
        return docs

    # 从文本中获取文档
    def get_docs_from_text(self, file_path):
        loader = TextLoader(file_path)
        docs = self.split_document(loader)
        print(Fore.LIGHTRED_EX + f"{file_path}: fetched {len(docs)} docs" + Style.RESET_ALL)
        return docs

    # 从pdf中获取文档
    def get_docs_from_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        docs = self.split_document(loader)
        print(Fore.LIGHTRED_EX + f"{file_path}:fetched {len(docs)} docs" + Style.RESET_ALL)
        return docs

    # 从文件中获取文档
    def get_docs_from_file(self, file_path):
        loader = UnstructuredFileLoader(file_path)
        docs = self.split_document(loader)
        print(Fore.LIGHTRED_EX + f"{file_path}:fetched {len(docs)} docs" + Style.RESET_ALL)
        return docs

    def get_docs_from(self, file_path):
        if file_path.endswith(".txt"):
            docs = self.get_docs_from_text(file_path)
        elif file_path.endswith(".pdf"):
            docs = self.get_docs_from_pdf(file_path)
        else:
            docs = self.get_docs_from_file(file_path)
        return docs

    def get_files_from_directory(self, directory_path, file_type="pdf"):
        files = []
        # 循环遍历目录下的所有文件，返回文件名
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file_type and not file.endswith(file_type):
                    continue
                file_path = os.path.join(root, file)
                logging.warning(f"Load file_path: {file_path}")
                yield file_path

    # 从目录中获取文档
    def get_docs_from_directory(self, directory_path, file_type="pdf"):
        files = self.get_files_from_directory(directory_path, file_type)
        for file in files:
            print(f"Load file_path: {file}")
            docs = self.get_docs_from(file)
            yield docs