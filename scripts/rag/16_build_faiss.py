import numpy as np
import faiss

from pathlib import Path


# =====================================================
# 路径
# =====================================================

# 当前文件：
# 项目根目录/scripts/16_build_faiss.py
BASE_DIR = Path(__file__).resolve().parents[2]


EMBED_DIR = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "embedding"
)


EMBED_FILE = (
    EMBED_DIR
    /
    "embeddings.npy"
)


METADATA_FILE = (
    EMBED_DIR
    /
    "metadata.json"
)


INDEX_FILE = (
    EMBED_DIR
    /
    "cnt_faiss.index"
)


# =====================================================
# 检查文件
# =====================================================

if not EMBED_FILE.exists():

    raise FileNotFoundError(
        f"找不到embedding文件: {EMBED_FILE}"
    )


if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"找不到metadata文件: {METADATA_FILE}"
    )


# =====================================================
# 加载embedding
# =====================================================

print(
    "加载embedding..."
)


embeddings = np.load(
    EMBED_FILE
)


# FAISS使用float32
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
# 建立FAISS索引
# =====================================================

dimension = embeddings.shape[1]


print(
    "向量维度:",
    dimension
)


# 15_build_embedding.py 已经使用：
# normalize_embeddings=True
#
# 因此这里使用 Inner Product
# 等价于 cosine similarity
index = faiss.IndexFlatIP(
    dimension
)


print(
    "加入向量..."
)


index.add(
    embeddings
)


print(
    "向量数量:",
    index.ntotal
)


# =====================================================
# 保存index
# =====================================================

faiss.write_index(
    index,
    str(INDEX_FILE)
)


# =====================================================
# 完成
# =====================================================

print(
    "==================="
)


print(
    "FAISS建立完成"
)


print(
    "索引文件:"
)


print(
    INDEX_FILE
)


print(
    "向量数量:",
    index.ntotal
)


print(
    "向量维度:",
    dimension
)


print(
    "==================="
)