import json
from pathlib import Path

import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


# =====================================================
# 路径
# =====================================================

# 当前文件：
# 项目根目录/scripts/17_retriever.py
BASE_DIR = Path(__file__).resolve().parents[2]


# BGE模型
MODEL_PATH = (
    BASE_DIR
    /
    "models"
    /
    "bge-base-en-v1.5"
)


# FAISS索引
INDEX_FILE = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
    /
    "cnt_faiss.index"
)


# metadata
METADATA_FILE = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
    /
    "metadata.json"
)


# =====================================================
# 检查文件
# =====================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"找不到BGE模型: {MODEL_PATH}"
    )


if not INDEX_FILE.exists():

    raise FileNotFoundError(
        f"找不到FAISS索引: {INDEX_FILE}"
    )


if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"找不到metadata: {METADATA_FILE}"
    )


# =====================================================
# device
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "================================="
)

print(
    "YanlongGPT-CNT Retriever"
)

print(
    "================================="
)


print(
    "device:",
    device
)


# =====================================================
# 加载BGE
# =====================================================

print(
    "\n加载embedding模型..."
)


embed_model = SentenceTransformer(
    str(MODEL_PATH),
    device=device
)


embed_model.eval()


print(
    "embedding模型加载完成"
)


# =====================================================
# 加载FAISS
# =====================================================

print(
    "\n加载FAISS索引..."
)


index = faiss.read_index(
    str(INDEX_FILE)
)


print(
    "FAISS向量数量:",
    index.ntotal
)


# =====================================================
# 加载metadata
# =====================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)


print(
    "metadata数量:",
    len(metadata)
)


# FAISS向量和metadata应该一一对应
if index.ntotal != len(metadata):

    raise ValueError(
        f"FAISS向量数量({index.ntotal}) "
        f"与metadata数量({len(metadata)})不一致"
    )


print(
    "metadata加载完成"
)


# =====================================================
# 检索函数
# =====================================================

def retrieve(
    query,
    top_k=5
):

    """
    输入:
        query:
            用户问题

        top_k:
            返回多少个相关chunk

    输出:
        最相关chunk的检索信息
    """


    # -------------------------
    # 1. query embedding
    # -------------------------

    query_vector = embed_model.encode(

        [query],

        normalize_embeddings=True,

        convert_to_numpy=True

    )


    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )


    # -------------------------
    # 2. FAISS搜索
    # -------------------------

    scores, ids = index.search(
        query_vector,
        top_k
    )


    results = []


    # -------------------------
    # 3. 找回metadata
    # -------------------------

    for score, idx in zip(
        scores[0],
        ids[0]
    ):


        if idx == -1:

            continue


        item = metadata[idx]


        results.append(

            {

                "score":
                    float(score),

                "id":
                    item["id"],

                "source":
                    item["source"],

                "chunk_id":
                    item["chunk_id"]

            }

        )


    return results


# =====================================================
# 测试
# =====================================================

if __name__ == "__main__":


    print(
        "\n=============================="
    )

    print(
        "Retriever测试"
    )

    print(
        "输入 exit 退出"
    )

    print(
        "=============================="
    )


    while True:


        query = input(
            "\nQuestion:\n> "
        ).strip()


        if query.lower() in [
            "exit",
            "quit",
            "q"
        ]:

            break


        if not query:

            continue


        results = retrieve(
            query,
            top_k=5
        )


        print(
            "\n========== Retrieved =========="
        )


        for i, r in enumerate(
            results
        ):


            print(
                f"\nRank {i + 1}"
            )


            print(
                "score:",
                r["score"]
            )


            print(
                "source:",
                r["source"]
            )


            print(
                "chunk:",
                r["chunk_id"]
            )


        print(
            "\n=============================="
        )