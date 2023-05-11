import wget
import pathlib
from pinecone_text.sparse import BM25Encoder

# 导入Pinecone数据库需要的bm25_encoder
def load_bm25(path: str):

    # 每次下载文件
    # bm25_encoder = BM25Encoder().default()

    # 替代方案
    bm25_encoder = BM25Encoder()
    url = "https://storage.googleapis.com/pinecone-datasets-dev/bm25_params/msmarco_bm25_params_v4_0_0.json"
    p = pathlib.Path(path)
    if not p.exists():
        wget.download(url, str(p))
    bm25_encoder.load(str(p))
    return bm25_encoder


if __name__ == '__main__':
    bm25 = load_bm25("./bm25.json")