# 初始化 pinecone
import os
import pinecone
pinecone.init(
    api_key = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENVIRONMENT")
)
index_name = "langchain-pinecone-hybrid-search"

# 初始化bm25_encoder
from bm25 import load_bm25
bm25_encoder = load_bm25("./bm25.json")

# 初始化向量查询接口
from langchain.vectorstores import Pinecone
from langchain.retrievers import PineconeHybridSearchRetriever

def vector_store(embeddings):
    index = pinecone.Index(index_name)
    docsearch = Pinecone.from_existing_index(
        index_name,
        embeddings
        )
    if bm25_encoder:
        # use hybrid model
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index
            )
    else:
        # use simple model
        retriever = docsearch.as_retriever()

    return docsearch, retriever

if __name__ == "__main__":
    
    from config import config
    config()
    print("pinecone: ", pinecone.whoami())

    ## create the index
    # pinecone.create_index(
    #     index_name,
    #     dimension = 1536,  # dimensionality of dense model
    #     metric = "dotproduct",  # sparse values supported only for dotproduct
    #     pod_type = "Starer",
    #     metadata_config = {"indexed": []}  # see explaination above
    # )