import json
from pathlib import Path
import importlib.util

import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer


# =====================================================
# 路径
# 当前文件：
# 项目根目录/scripts/rag/18_chat_rag.py
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]


MODEL_CODE_FILE = (
    BASE_DIR
    / "scripts"
    / "06_yanlongllm.py"
)

MODEL_FILE = (
    BASE_DIR
    / "checkpoint_sft"
    / "yanlongllm_cnt_sft_epoch3.pt"
)

TOKENIZER_FILE = (
    BASE_DIR
    / "tokenizer"
    / "cnt_tokenizer.json"
)

BGE_PATH = (
    BASE_DIR
    / "models"
    / "bge-base-en-v1.5"
)

INDEX_FILE = (
    BASE_DIR
    / "data"
    / "rag"
    / "embedding"
    / "cnt_faiss.index"
)

METADATA_FILE = (
    BASE_DIR
    / "data"
    / "rag"
    / "embedding"
    / "metadata.json"
)

CHUNK_FILE = (
    BASE_DIR
    / "data"
    / "rag"
    / "clean_chunks.jsonl"
)


# =====================================================
# 参数
# =====================================================

RETRIEVE_TOP_K = 3

MAX_NEW_TOKENS = 200

TEMPERATURE = 0.3

TOP_K = 10

REPETITION_PENALTY = 1.2

SEED = 42


# =====================================================
# 文件检查
# =====================================================

for file_path, name in [

    (MODEL_CODE_FILE, "06_yanlongllm.py"),

    (MODEL_FILE, "SFT模型"),

    (TOKENIZER_FILE, "Tokenizer"),

    (BGE_PATH, "BGE模型"),

    (INDEX_FILE, "FAISS索引"),

    (METADATA_FILE, "metadata"),

    (CHUNK_FILE, "clean chunks"),

]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到{name}: {file_path}"
        )


# =====================================================
# 动态导入 06_yanlongllm.py
# =====================================================

spec_model = importlib.util.spec_from_file_location(
    "yanlongllm_module",
    MODEL_CODE_FILE
)

yanlongllm_module = importlib.util.module_from_spec(
    spec_model
)

spec_model.loader.exec_module(
    yanlongllm_module
)

YanlongLLM = yanlongllm_module.YanlongLLM
YanlongConfig = yanlongllm_module.YanlongConfig


# =====================================================
# device
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


print(
    "================================="
)

print(
    "YanlongGPT-CNT v2 RAG"
)

print(
    "================================="
)

print(
    "device:",
    device
)


# =====================================================
# tokenizer
# =====================================================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)

eos_id = tokenizer.token_to_id(
    "[EOS]"
)

print(
    "vocab:",
    tokenizer.get_vocab_size()
)

print(
    "EOS:",
    eos_id
)


# =====================================================
# 加载BGE
# =====================================================

print(
    "\n加载BGE..."
)

embed_model = SentenceTransformer(
    str(BGE_PATH),
    device=device
)

embed_model.eval()

print(
    "BGE加载完成"
)


# =====================================================
# 加载FAISS
# =====================================================

print(
    "\n加载FAISS..."
)

index = faiss.read_index(
    str(INDEX_FILE)
)

print(
    "FAISS向量数量:",
    index.ntotal
)


# =====================================================
# metadata
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


if index.ntotal != len(metadata):

    raise ValueError(
        f"FAISS向量数量({index.ntotal}) "
        f"与metadata数量({len(metadata)})不一致"
    )


# =====================================================
# chunk文本
# =====================================================

chunk_texts = {}


with open(
    CHUNK_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:

            continue

        item = json.loads(
            line
        )

        chunk_texts[
            item["id"]
        ] = item["text"]


print(
    "chunk加载完成:",
    len(chunk_texts)
)


# =====================================================
# 加载YanlongGPT
# =====================================================

print(
    "\n加载YanlongGPT..."
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)


if "config" not in checkpoint:

    raise KeyError(
        "SFT checkpoint中没有找到config"
    )


config = YanlongConfig(
    **checkpoint["config"]
)


print(
    "模型Config:"
)

print(
    "vocab_size:",
    config.vocab_size
)

print(
    "block_size:",
    config.block_size
)

print(
    "n_layer:",
    config.n_layer
)

print(
    "n_head:",
    config.n_head
)

print(
    "n_embd:",
    config.n_embd
)


model = YanlongLLM(
    config
)


if "model" not in checkpoint:

    raise KeyError(
        "SFT checkpoint中没有找到model参数"
    )


model.load_state_dict(
    checkpoint["model"]
)

model.to(
    device
)

model.eval()

del checkpoint


print(
    "YanlongGPT加载完成"
)


# =====================================================
# 检索
# =====================================================

def retrieve(
    query,
    top_k=RETRIEVE_TOP_K
):

    vector = embed_model.encode(

        [query],

        normalize_embeddings=True,

        convert_to_numpy=True

    )


    vector = np.asarray(
        vector,
        dtype=np.float32
    )


    scores, ids = index.search(
        vector,
        top_k
    )


    contexts = []


    for score, idx in zip(
        scores[0],
        ids[0]
    ):

        if idx == -1:

            continue


        meta = metadata[idx]


        text = chunk_texts.get(
            meta["id"],
            ""
        )


        if not text:

            continue


        contexts.append(
            {
                "text":
                    text,

                "score":
                    float(score),

                "source":
                    meta["source"],

                "chunk_id":
                    meta["chunk_id"],

                "used_tokens":
                    0
            }
        )


    return contexts


# =====================================================
# 构建RAG Prompt
#
# 重要：
# 保持SFT训练时的外层格式：
#
# ### Instruction:
# ...
#
# ### Response:
#
# 不再额外使用 ### Context:
# =====================================================

def build_rag_prompt_ids(
    question,
    contexts,
    max_prompt_length
):

    instruction_header = (
        "### Instruction:\n"
        "Use the following retrieved evidence to answer "
        "the CNT question accurately."
    )


    question_tail = (
        "\n\nQuestion:\n"
        + question
        + "\n\n### Response:\n"
    )


    header_ids = tokenizer.encode(
        instruction_header
    ).ids


    tail_ids = tokenizer.encode(
        question_tail
    ).ids


    # -------------------------------------------------
    # 如果问题本身极端过长，优先保留：
    # Instruction格式 + Question + Response标记
    # -------------------------------------------------

    if (
        len(header_ids)
        +
        len(tail_ids)
        >=
        max_prompt_length
    ):

        simple_header = (
            "### Instruction:\n"
        )

        simple_tail = (
            "\n\n### Response:\n"
        )

        simple_header_ids = tokenizer.encode(
            simple_header
        ).ids

        simple_tail_ids = tokenizer.encode(
            simple_tail
        ).ids

        question_ids = tokenizer.encode(
            question
        ).ids

        available_question_tokens = (
            max_prompt_length
            -
            len(simple_header_ids)
            -
            len(simple_tail_ids)
        )

        if available_question_tokens < 1:

            raise ValueError(
                "block_size过小，无法构建Prompt"
            )

        question_ids = question_ids[
            :available_question_tokens
        ]

        prompt_ids = (
            simple_header_ids
            +
            question_ids
            +
            simple_tail_ids
        )

        return prompt_ids, contexts


    # -------------------------------------------------
    # 正常情况：
    # 从Rank 1开始按顺序加入Evidence
    #
    # 永远优先Rank 1
    # 空间不够时截断后面的Evidence
    # -------------------------------------------------

    prompt_ids = list(
        header_ids
    )


    for i, item in enumerate(
        contexts,
        start=1
    ):

        evidence_label = (
            f"\n\nEvidence {i}:\n"
        )

        label_ids = tokenizer.encode(
            evidence_label
        ).ids


        text_ids = tokenizer.encode(
            item["text"]
        ).ids


        remaining_tokens = (
            max_prompt_length
            -
            len(prompt_ids)
            -
            len(tail_ids)
        )


        # 连Evidence标签都放不下时停止
        if remaining_tokens <= len(label_ids):

            break


        prompt_ids.extend(
            label_ids
        )


        remaining_for_text = (
            remaining_tokens
            -
            len(label_ids)
        )


        # 当前Evidence完整放得下
        if len(text_ids) <= remaining_for_text:

            prompt_ids.extend(
                text_ids
            )

            item["used_tokens"] = len(
                text_ids
            )


        # 当前Evidence只能部分放入
        else:

            text_ids = text_ids[
                :remaining_for_text
            ]

            prompt_ids.extend(
                text_ids
            )

            item["used_tokens"] = len(
                text_ids
            )

            break


    # 最后加入Question和Response标记
    prompt_ids.extend(
        tail_ids
    )


    # 理论检查
    if len(prompt_ids) > max_prompt_length:

        raise RuntimeError(
            "RAG Prompt长度控制失败"
        )


    return prompt_ids, contexts


# =====================================================
# repetition penalty
#
# 只惩罚模型已经生成的回答token
# 不惩罚问题和RAG上下文
# =====================================================

def apply_repetition_penalty(
    logits,
    generated_token_ids,
    penalty
):

    if penalty <= 1.0:

        return logits


    unique_tokens = set(
        generated_token_ids
    )


    for token_id in unique_tokens:

        if token_id < 0:

            continue

        if token_id >= logits.size(-1):

            continue


        if logits[
            0,
            token_id
        ] < 0:

            logits[
                0,
                token_id
            ] *= penalty

        else:

            logits[
                0,
                token_id
            ] /= penalty


    return logits


# =====================================================
# Top-k Sampling
# =====================================================

def sample_next_token(
    logits,
    temperature=TEMPERATURE,
    top_k=TOP_K
):

    if temperature <= 0:

        raise ValueError(
            "temperature必须大于0"
        )


    logits = logits / temperature


    if (
        top_k is not None
        and
        top_k > 0
    ):

        top_k = min(
            top_k,
            logits.size(-1)
        )


        values, _ = torch.topk(
            logits,
            top_k
        )


        cutoff = values[
            :,
            -1
        ].unsqueeze(-1)


        logits = torch.where(

            logits < cutoff,

            torch.full_like(
                logits,
                float("-inf")
            ),

            logits

        )


    probs = torch.softmax(
        logits,
        dim=-1
    )


    next_token = torch.multinomial(
        probs,
        num_samples=1
    )


    return next_token


# =====================================================
# 生成
# =====================================================

@torch.no_grad()
def generate(
    question,
    max_new_tokens=MAX_NEW_TOKENS
):

    contexts = retrieve(
        question,
        top_k=RETRIEVE_TOP_K
    )


    # -------------------------------------------------
    # 给答案预留生成空间
    # -------------------------------------------------

    max_prompt_length = (
        config.block_size
        -
        max_new_tokens
    )


    if max_prompt_length <= 0:

        raise ValueError(
            "MAX_NEW_TOKENS不能大于等于block_size"
        )


    # -------------------------------------------------
    # 构建Prompt
    #
    # 与原代码最大的区别：
    # 不再先拼完整Context然后从左边截断
    #
    # 而是：
    # Rank1 -> Rank2 -> Rank3
    # 按剩余token预算依次加入
    # -------------------------------------------------

    prompt_ids, contexts = build_rag_prompt_ids(

        question=question,

        contexts=contexts,

        max_prompt_length=max_prompt_length

    )


    prompt_length = len(
        prompt_ids
    )


    input_ids = torch.tensor(

        prompt_ids,

        dtype=torch.long

    ).unsqueeze(0).to(
        device
    )


    # =================================================
    # 自回归生成
    # =================================================

    for _ in range(
        max_new_tokens
    ):


        idx = input_ids[
            :,
            -config.block_size:
        ]


        logits, _ = model(
            idx
        )


        logits = logits[
            :,
            -1,
            :
        ]


        # ---------------------------------------------
        # 只获取已经生成的答案token
        #
        # 不把：
        # Question
        # Evidence
        # Instruction
        # 算入 repetition penalty
        # ---------------------------------------------

        generated_token_ids = input_ids[
            0,
            prompt_length:
        ].tolist()


        logits = apply_repetition_penalty(

            logits,

            generated_token_ids,

            REPETITION_PENALTY

        )


        # ---------------------------------------------
        # Top-k + temperature sampling
        # 不再使用greedy argmax
        # ---------------------------------------------

        next_token = sample_next_token(

            logits,

            temperature=TEMPERATURE,

            top_k=TOP_K

        )


        # EOS停止
        if (
            eos_id is not None
            and
            next_token.item() == eos_id
        ):

            break


        input_ids = torch.cat(

            [
                input_ids,
                next_token
            ],

            dim=1

        )


    # =================================================
    # 只保留模型新生成的答案
    # =================================================

    output_ids = input_ids[
        0
    ].tolist()[
        prompt_length:
    ]


    answer = tokenizer.decode(
        output_ids
    )


    return (
        answer.strip(),
        contexts,
        prompt_length
    )


# =====================================================
# chat
# =====================================================

if __name__ == "__main__":


    print(
        "\n==================="
    )

    print(
        "YanlongGPT-CNT v2 RAG"
    )

    print(
        "输入 exit / quit / q 退出"
    )

    print(
        "==================="
    )


    while True:


        question = input(
            "\nUser:\n> "
        ).strip()


        if question.lower() in [

            "exit",

            "quit",

            "q"

        ]:

            break


        if not question:

            continue


        answer, contexts, prompt_length = generate(
            question
        )


        print(
            "\n========== Retrieved =========="
        )


        for i, item in enumerate(
            contexts
        ):


            print(
                f"\nRank {i + 1}"
            )


            print(
                "score:",
                item["score"]
            )


            print(
                "source:",
                item["source"]
            )


            print(
                "chunk:",
                item["chunk_id"]
            )


            print(
                "used in prompt:",
                item["used_tokens"],
                "tokens"
            )


            print(
                "text:",
                item["text"][:300]
            )


        print(
            "\nPrompt tokens:",
            prompt_length,
            "/",
            config.block_size
        )


        print(
            "\n================================="
        )


        print(
            "\nYanlongGPT:"
        )


        print(
            answer
        )