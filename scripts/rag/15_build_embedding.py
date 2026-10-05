import json
from pathlib import Path

import torch
import numpy as np

from sentence_transformers import SentenceTransformer


# =====================================================
# 路径
# =====================================================

# 当前文件：
# 项目根目录/scripts/15_build_embedding.py
BASE_DIR = Path(__file__).resolve().parents[2]


# 14_clean_chunks.py 输出的清洗后 chunk
INPUT_FILE = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "clean_chunks.jsonl"
)


# embedding 保存位置
OUTPUT_DIR = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 参数
# =====================================================

# 本地 BGE embedding 模型
MODEL_NAME = (
    BASE_DIR
    /
    "models"
    /
    "bge-base-en-v1.5"
)


# RTX 4060 Laptop 8GB
BATCH_SIZE = 16


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
    "YanlongGPT-CNT RAG Embedding"
)

print(
    "================================="
)


print(
    "device:",
    device
)


# =====================================================
# 检查文件
# =====================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"找不到clean chunks文件: {INPUT_FILE}"
    )


if not MODEL_NAME.exists():

    raise FileNotFoundError(
        f"找不到本地BGE模型: {MODEL_NAME}"
    )


print(
    "输入chunk:"
)

print(
    INPUT_FILE
)


print(
    "\nEmbedding模型:"
)

print(
    MODEL_NAME
)


# =====================================================
# 加载embedding模型
# =====================================================

print(
    "\n加载embedding模型..."
)


model = SentenceTransformer(
    str(MODEL_NAME),
    device=device
)


model.eval()


print(
    "模型加载完成"
)


# =====================================================
# 读取chunk
# =====================================================

texts = []

metadata = []


print(
    "\n读取chunks..."
)


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:


    for idx, line in enumerate(f):


        line = line.strip()


        if not line:

            continue


        item = json.loads(
            line
        )


        # 必须包含text字段
        if "text" not in item:

            continue


        text = item[
            "text"
        ].strip()


        # 空文本跳过
        if not text:

            continue


        texts.append(
            text
        )


        metadata.append(

            {

                "id":
                    item.get(
                        "id",
                        idx
                    ),

                "source":
                    item.get(
                        "source",
                        ""
                    ),

                "chunk_id":
                    item.get(
                        "chunk_id",
                        idx
                    )

            }

        )


print(
    "chunk数量:",
    len(texts)
)


if len(texts) == 0:

    raise ValueError(
        "没有可用于embedding的chunk"
    )


# =====================================================
# embedding
# =====================================================

print(
    "\n开始embedding..."
)


embeddings = model.encode(

    texts,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    normalize_embeddings=True,

    convert_to_numpy=True

)


# 确保FAISS后续使用float32
embeddings = embeddings.astype(
    np.float32
)


print(
    "embedding shape:",
    embeddings.shape
)


print(
    "embedding dtype:",
    embeddings.dtype
)


# =====================================================
# 保存embedding
# =====================================================

embedding_file = (
    OUTPUT_DIR
    /
    "embeddings.npy"
)


np.save(
    embedding_file,
    embeddings
)


# =====================================================
# 保存metadata
# =====================================================

metadata_file = (
    OUTPUT_DIR
    /
    "metadata.json"
)


with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as f:


    json.dump(

        metadata,

        f,

        ensure_ascii=False,

        indent=2

    )


# =====================================================
# 完成
# =====================================================

print(
    "\n=============================="
)


print(
    "embedding完成"
)


print(
    "chunk数量:",
    len(texts)
)


print(
    "向量维度:",
    embeddings.shape[1]
)


print(
    "向量文件:"
)

print(
    embedding_file
)


print(
    "metadata:"
)

print(
    metadata_file
)


print(
    "=============================="
)