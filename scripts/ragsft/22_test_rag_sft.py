import json
import re
import importlib.util
from pathlib import Path

import torch
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer


# =====================================================
# 路径
#
# 当前文件：
# 项目根目录/scripts/ragsft/22_test_rag_sft.py
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]


MODEL_CODE_FILE = (
    BASE_DIR
    / "scripts"
    / "06_yanlongllm.py"
)


# 普通SFT基线
SFT_MODEL_FILE = (
    BASE_DIR
    / "checkpoint_sft"
    / "yanlongllm_cnt_sft_epoch3.pt"
)


# RAG-aware SFT最终测试模型
#
# 默认使用训练脚本根据val loss保存的best checkpoint。
RAGSFT_MODEL_FILE = (
    BASE_DIR
    / "checkpoint_ragsft"
    / "yanlongllm_cnt_ragsft_best.pt"
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


OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "rag"
    / "eval"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "rag_sft_comparison.jsonl"
)


# =====================================================
# 参数
# =====================================================

# RAG-aware SFT训练时最多使用1-2个Evidence chunk。
# 测试保持同一分布。
RETRIEVE_TOP_K = 2


# 模型block_size=1024。
#
# 第一版220 token容易把22M小模型推入长重复循环。
# 这里缩短到160 token，优先观察模型能否给出更短、
# 更聚焦、真正基于Evidence的回答。
MAX_NEW_TOKENS = 160


# =====================================================
# Deterministic anti-repetition decoding
#
# 不使用随机sampling，仍然是可复现的greedy比较；
# 但在argmax之前加入：
#
# 1. repetition penalty
# 2. no-repeat n-gram
# 3. 特殊token屏蔽
#
# 普通SFT与RAG-SFT使用完全相同的解码策略，
# 因此仍然是受控比较。
# =====================================================

REPETITION_PENALTY = 1.18

NO_REPEAT_NGRAM_SIZE = 4

# 只对最近生成的token施加repetition penalty，
# 避免Evidence里已经出现过的科学术语被错误惩罚。
REPETITION_WINDOW = 96

DECODING_STRATEGY = "penalized_greedy"


# 是否运行固定诊断问题
RUN_FIXED_TESTS = True


# 固定问题跑完以后是否继续进入交互模式
RUN_INTERACTIVE = True


# =====================================================
# 固定诊断问题
#
# 这些问题用于复现之前普通SFT + RAG表现较弱的场景。
# =====================================================

TEST_QUESTIONS = [

    (
        "How does catalyst particle size influence carbon nanotube "
        "diameter and diameter distribution during CVD growth?"
    ),

    (
        "What information do the Raman D and G bands provide about "
        "the structural quality and defect state of carbon nanotubes?"
    ),

    (
        "How does growth temperature affect carbon nanotube growth, "
        "morphology, and structural quality during chemical vapor deposition?"
    ),

    (
        "How can carbon nanotube diameter be controlled through catalyst "
        "properties and CVD process conditions?"
    ),

]


# =====================================================
# Prompt
#
# 必须与21_rag_sft_dataset.py训练格式一致：
#
# ### Instruction:
# Use the following retrieved evidence to answer ...
#
# Evidence 1:
# ...
#
# Evidence 2:
# ...
#
# Question:
# ...
#
# ### Response:
# =====================================================

INSTRUCTION_HEADER = (
    "### Instruction:\n"
    "Use the following retrieved evidence to answer "
    "the CNT question accurately."
)

QUESTION_HEADER = (
    "\n\nQuestion:\n"
)

RESPONSE_HEADER = (
    "\n\n### Response:\n"
)


def build_evidence_header(
    index
):
    return (
        f"\n\nEvidence {index}:\n"
    )


# =====================================================
# 文件检查
# =====================================================

for path, name in [

    (
        MODEL_CODE_FILE,
        "06_yanlongllm.py"
    ),

    (
        SFT_MODEL_FILE,
        "普通SFT checkpoint"
    ),

    (
        RAGSFT_MODEL_FILE,
        "RAG-SFT best checkpoint"
    ),

    (
        TOKENIZER_FILE,
        "Tokenizer"
    ),

    (
        BGE_PATH,
        "BGE模型"
    ),

    (
        INDEX_FILE,
        "FAISS索引"
    ),

    (
        METADATA_FILE,
        "metadata"
    ),

    (
        CHUNK_FILE,
        "clean chunks"
    ),

]:

    if not path.exists():

        raise FileNotFoundError(
            f"找不到{name}: {path}"
        )


# =====================================================
# 动态导入模型
# =====================================================

spec_model = importlib.util.spec_from_file_location(
    "yanlongllm_module",
    MODEL_CODE_FILE
)

yanlongllm_module = (
    importlib.util.module_from_spec(
        spec_model
    )
)

spec_model.loader.exec_module(
    yanlongllm_module
)

YanlongLLM = (
    yanlongllm_module.YanlongLLM
)

YanlongConfig = (
    yanlongllm_module.YanlongConfig
)


# =====================================================
# Device
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


print(
    "================================="
)

print(
    "YanlongGPT-CNT v2 RAG-SFT Test"
)

print(
    "================================="
)

print(
    "device:",
    device
)

print(
    "retrieve top_k:",
    RETRIEVE_TOP_K
)

print(
    "max_new_tokens:",
    MAX_NEW_TOKENS
)


# =====================================================
# Tokenizer
# =====================================================

tokenizer = Tokenizer.from_file(
    str(
        TOKENIZER_FILE
    )
)


eos_id = tokenizer.token_to_id(
    "[EOS]"
)


pad_id = tokenizer.token_to_id(
    "[PAD]"
)


bos_id = tokenizer.token_to_id(
    "[BOS]"
)


unk_id = tokenizer.token_to_id(
    "[UNK]"
)


print(
    "vocab:",
    tokenizer.get_vocab_size()
)

print(
    "EOS:",
    eos_id
)

print(
    "PAD:",
    pad_id
)


print(
    "BOS:",
    bos_id
)


print(
    "UNK:",
    unk_id
)


print(
    "decoding:",
    DECODING_STRATEGY
)


print(
    "repetition_penalty:",
    REPETITION_PENALTY
)


print(
    "no_repeat_ngram_size:",
    NO_REPEAT_NGRAM_SIZE
)


print(
    "repetition_window:",
    REPETITION_WINDOW
)


# =====================================================
# BGE
# =====================================================

print(
    "\n加载BGE..."
)


embed_model = SentenceTransformer(
    str(
        BGE_PATH
    ),
    device=device
)


embed_model.eval()


print(
    "BGE加载完成"
)


# =====================================================
# FAISS
# =====================================================

print(
    "\n加载FAISS..."
)


index = faiss.read_index(
    str(
        INDEX_FILE
    )
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

    encoding="utf-8",

) as f:

    metadata = json.load(
        f
    )


if len(
    metadata
) != index.ntotal:

    raise RuntimeError(
        "metadata数量与FAISS向量数量不一致: "
        f"{len(metadata)} != {index.ntotal}"
    )


print(
    "metadata数量:",
    len(
        metadata
    )
)


# =====================================================
# chunk文本
# =====================================================

chunk_texts = {}


with open(

    CHUNK_FILE,

    "r",

    encoding="utf-8",

) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        item = json.loads(
            line
        )

        chunk_id = str(
            item.get(
                "id",
                ""
            )
        ).strip()

        text = str(
            item.get(
                "text",
                ""
            )
        ).strip()

        if (
            chunk_id
            and
            text
        ):

            chunk_texts[
                chunk_id
            ] = text


print(
    "chunk加载完成:",
    len(
        chunk_texts
    )
)


# =====================================================
# 加载模型
# =====================================================

def load_model(
    checkpoint_file,
    label,
):

    print(
        f"\n加载{label}:"
    )

    print(
        checkpoint_file
    )

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    if "model" not in checkpoint:

        raise KeyError(
            f"{label} checkpoint中没有model"
        )

    if "config" not in checkpoint:

        raise KeyError(
            f"{label} checkpoint中没有config"
        )

    config = YanlongConfig(
        **checkpoint[
            "config"
        ]
    )

    model = YanlongLLM(
        config
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    model.to(
        device
    )

    model.eval()

    parameter_count = sum(

        p.numel()

        for p in model.parameters()

    )

    print(
        f"{label}加载完成"
    )

    print(
        "参数量:",
        parameter_count / 1e6,
        "M"
    )

    print(
        "block_size:",
        config.block_size
    )

    extra_info = {

        "epoch":
            checkpoint.get(
                "epoch"
            ),

        "train_loss":
            checkpoint.get(
                "train_loss",
                checkpoint.get(
                    "sft_loss"
                )
            ),

        "val_loss":
            checkpoint.get(
                "val_loss"
            ),

        "training_stage":
            checkpoint.get(
                "training_stage"
            ),

    }

    del checkpoint

    return (
        model,
        config,
        extra_info
    )


sft_model, sft_config, sft_info = (
    load_model(
        SFT_MODEL_FILE,
        "普通SFT模型"
    )
)


ragsft_model, ragsft_config, ragsft_info = (
    load_model(
        RAGSFT_MODEL_FILE,
        "RAG-SFT模型"
    )
)


# =====================================================
# Config一致性检查
# =====================================================

important_config_fields = [

    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "n_embd",

]


for field in important_config_fields:

    baseline_value = getattr(
        sft_config,
        field
    )

    ragsft_value = getattr(
        ragsft_config,
        field
    )

    if baseline_value != ragsft_value:

        raise RuntimeError(
            "普通SFT与RAG-SFT模型结构不一致: "
            f"{field}: "
            f"{baseline_value} != {ragsft_value}"
        )


if (
    tokenizer.get_vocab_size()
    !=
    ragsft_config.vocab_size
):

    raise RuntimeError(
        "Tokenizer vocab与模型vocab不一致"
    )


print(
    "\n模型结构一致性检查通过"
)


print(
    "普通SFT信息:",
    sft_info
)


print(
    "RAG-SFT信息:",
    ragsft_info
)


# =====================================================
# 检索
# =====================================================

@torch.no_grad()
def retrieve(
    query,
    top_k=RETRIEVE_TOP_K,
):

    vector = embed_model.encode(

        [
            query
        ],

        normalize_embeddings=True,

        convert_to_numpy=True,

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

        scores[
            0
        ],

        ids[
            0
        ],

    ):

        if idx < 0:
            continue

        meta = metadata[
            int(
                idx
            )
        ]


        global_chunk_id = str(
            meta.get(
                "id",
                ""
            )
        ).strip()


        text = chunk_texts.get(
            global_chunk_id,
            ""
        )


        if not text:
            continue


        contexts.append(
            {

                "text":
                    text,

                "score":
                    float(
                        score
                    ),

                "source":
                    str(
                        meta.get(
                            "source",
                            ""
                        )
                    ),

                "chunk_id":
                    str(
                        meta.get(
                            "chunk_id",
                            global_chunk_id
                        )
                    ),

                "id":
                    global_chunk_id,

            }
        )


    return contexts


# =====================================================
# Prompt构建
# =====================================================

def build_prompt_text(
    question,
    evidence_texts,
):

    parts = [
        INSTRUCTION_HEADER
    ]


    for evidence_index, evidence_text in enumerate(

        evidence_texts,

        start=1,

    ):

        parts.append(
            build_evidence_header(
                evidence_index
            )
        )

        parts.append(
            evidence_text
        )


    parts.append(
        QUESTION_HEADER
    )

    parts.append(
        question
    )

    parts.append(
        RESPONSE_HEADER
    )


    return "".join(
        parts
    )


# =====================================================
# Evidence公平截断
#
# 推理时必须给生成答案预留MAX_NEW_TOKENS，
# 因此Prompt最多：
#
# block_size - MAX_NEW_TOKENS
#
# 不截Question，不截Instruction，不截Response header，
# 只压缩Evidence。
# =====================================================

def allocate_evidence_tokens(
    token_lists,
    total_budget,
):

    n = len(
        token_lists
    )


    if n == 0:
        return []


    lengths = [

        len(
            tokens
        )

        for tokens in token_lists

    ]


    if sum(
        lengths
    ) <= total_budget:

        return lengths


    # 每个Evidence至少保留16 token
    minimums = [

        min(
            16,
            length
        )

        for length in lengths

    ]


    minimum_total = sum(
        minimums
    )


    if minimum_total > total_budget:

        raise RuntimeError(
            "Prompt预算过小，无法保留必要Evidence"
        )


    allocations = list(
        minimums
    )


    remaining = (
        total_budget
        -
        minimum_total
    )


    capacities = [

        length
        -
        minimum

        for length, minimum in zip(

            lengths,

            minimums,

        )

    ]


    total_capacity = sum(
        capacities
    )


    if (
        remaining > 0
        and
        total_capacity > 0
    ):

        proportional = []


        for capacity in capacities:

            share = int(

                remaining
                *
                capacity
                /
                total_capacity

            )

            proportional.append(

                min(
                    share,
                    capacity
                )

            )


        for i, share in enumerate(
            proportional
        ):

            allocations[
                i
            ] += share


        leftover = (
            total_budget
            -
            sum(
                allocations
            )
        )


        while leftover > 0:

            changed = False


            for i in range(
                n
            ):

                if (
                    allocations[
                        i
                    ]
                    <
                    lengths[
                        i
                    ]
                ):

                    allocations[
                        i
                    ] += 1

                    leftover -= 1

                    changed = True


                    if leftover == 0:
                        break


            if not changed:
                break


    return allocations


def build_prompt_ids(
    question,
    contexts,
    config,
    max_new_tokens=MAX_NEW_TOKENS,
):

    evidence_texts = [

        item[
            "text"
        ]

        for item in contexts

    ]


    if not evidence_texts:

        raise RuntimeError(
            "没有可用Retrieved Evidence"
        )


    max_prompt_tokens = (
        config.block_size
        -
        max_new_tokens
    )


    if max_prompt_tokens <= 64:

        raise ValueError(
            "max_new_tokens过大，留给Prompt的空间不足"
        )


    # -------------------------------------------------
    # 先尝试完整Evidence
    # -------------------------------------------------

    full_prompt_text = build_prompt_text(
        question,
        evidence_texts
    )


    full_prompt_ids = tokenizer.encode(
        full_prompt_text
    ).ids


    if len(
        full_prompt_ids
    ) <= max_prompt_tokens:

        return (
            full_prompt_ids,
            full_prompt_text,
            False,
        )


    # -------------------------------------------------
    # Evidence需要压缩
    # -------------------------------------------------

    empty_evidence_texts = [

        ""

        for _ in evidence_texts

    ]


    base_prompt_text = build_prompt_text(
        question,
        empty_evidence_texts
    )


    base_prompt_ids = tokenizer.encode(
        base_prompt_text
    ).ids


    evidence_budget = (
        max_prompt_tokens
        -
        len(
            base_prompt_ids
        )
    )


    if evidence_budget <= 0:

        raise RuntimeError(
            "Question和固定Prompt已经超过推理预算"
        )


    evidence_token_lists = [

        tokenizer.encode(
            text
        ).ids

        for text in evidence_texts

    ]


    allocations = allocate_evidence_tokens(
        evidence_token_lists,
        evidence_budget
    )


    def rebuild(
        current_allocations
    ):

        current_texts = []


        for tokens, allocation in zip(

            evidence_token_lists,

            current_allocations,

        ):

            kept = tokens[
                :allocation
            ]


            text = tokenizer.decode(
                kept,
                skip_special_tokens=False
            ).strip()


            current_texts.append(
                text
            )


        prompt_text = build_prompt_text(
            question,
            current_texts
        )


        prompt_ids = tokenizer.encode(
            prompt_text
        ).ids


        return (
            prompt_ids,
            prompt_text
        )


    prompt_ids, prompt_text = rebuild(
        allocations
    )


    # -------------------------------------------------
    # 边界token化差异安全收缩
    # -------------------------------------------------

    while len(
        prompt_ids
    ) > max_prompt_tokens:

        reducible = [

            allocations[
                i
            ]
            -
            min(
                16,
                len(
                    evidence_token_lists[
                        i
                    ]
                )
            )

            for i in range(
                len(
                    allocations
                )
            )

        ]


        max_reducible = max(
            reducible
        )


        if max_reducible <= 0:

            raise RuntimeError(
                "Evidence压缩到最小仍超过Prompt预算"
            )


        reduce_index = reducible.index(
            max_reducible
        )


        overflow = (
            len(
                prompt_ids
            )
            -
            max_prompt_tokens
        )


        reduction = min(

            max(
                1,
                overflow + 2
            ),

            max_reducible,

        )


        allocations[
            reduce_index
        ] -= reduction


        prompt_ids, prompt_text = rebuild(
            allocations
        )


    return (
        prompt_ids,
        prompt_text,
        True,
    )


# =====================================================
# Anti-repetition helpers
# =====================================================

def apply_repetition_penalty(
    logits,
    generated_ids,
    penalty=REPETITION_PENALTY,
    window=REPETITION_WINDOW,
):
    """
    HuggingFace风格的deterministic repetition penalty。

    只惩罚已经生成过的token，不惩罚Prompt/Evidence中的token。
    这样不会因为某个术语出现在Evidence里，就禁止模型正常使用它。
    """

    if (
        penalty is None
        or penalty <= 1.0
        or not generated_ids
    ):
        return logits


    if (
        window is not None
        and window > 0
    ):
        recent_ids = generated_ids[
            -window:
        ]
    else:
        recent_ids = generated_ids


    # 同一个token只处理一次。
    token_ids = set(
        int(token_id)
        for token_id in recent_ids
    )


    for token_id in token_ids:

        if (
            token_id < 0
            or token_id >= logits.shape[-1]
        ):
            continue


        token_logit = logits[
            0,
            token_id
        ]


        # 标准repetition penalty：
        # 正logit除以penalty；
        # 负logit乘以penalty。
        if token_logit.item() < 0:

            logits[
                0,
                token_id
            ] = (
                token_logit
                *
                penalty
            )

        else:

            logits[
                0,
                token_id
            ] = (
                token_logit
                /
                penalty
            )


    return logits


def get_banned_ngram_tokens(
    generated_ids,
    ngram_size=NO_REPEAT_NGRAM_SIZE,
):
    """
    返回当前step禁止生成的token，
    防止已经出现过的n-gram再次完整出现。

    只基于模型新生成的答案，不把Prompt/Evidence算进去。
    """

    if (
        ngram_size is None
        or ngram_size <= 0
    ):
        return set()


    if len(
        generated_ids
    ) + 1 < ngram_size:

        return set()


    generated_ngrams = {}


    for i in range(
        len(generated_ids)
        -
        ngram_size
        +
        1
    ):

        ngram = generated_ids[
            i:
            i + ngram_size
        ]

        prefix = tuple(
            ngram[
                :-1
            ]
        )

        next_token = int(
            ngram[
                -1
            ]
        )

        if prefix not in generated_ngrams:

            generated_ngrams[
                prefix
            ] = set()


        generated_ngrams[
            prefix
        ].add(
            next_token
        )


    current_prefix = tuple(
        generated_ids[
            -(
                ngram_size
                -
                1
            ):
        ]
    )


    return generated_ngrams.get(
        current_prefix,
        set()
    )


def block_invalid_special_tokens(
    logits,
):
    """
    生成阶段禁止PAD/BOS/UNK。
    EOS仍然保留，允许模型自然结束答案。
    """

    blocked_ids = [

        pad_id,

        bos_id,

        unk_id,

    ]


    for token_id in blocked_ids:

        if token_id is None:
            continue

        if (
            0
            <= token_id
            < logits.shape[-1]
        ):

            logits[
                0,
                token_id
            ] = -float(
                "inf"
            )


    return logits


# =====================================================
# 生成
# =====================================================

@torch.no_grad()
def generate_from_model(
    model,
    config,
    prompt_ids,
    max_new_tokens=MAX_NEW_TOKENS,
):

    input_ids = torch.tensor(

        prompt_ids,

        dtype=torch.long,

    ).unsqueeze(
        0
    ).to(
        device
    )


    prompt_length = len(
        prompt_ids
    )


    # 这里只记录“新生成答案”的token。
    #
    # repetition penalty / no-repeat ngram
    # 都只基于这些token，
    # 不基于Prompt和Evidence。
    generated_ids = []


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


        next_logits = logits[
            :,
            -1,
            :
        ].clone()


        # =================================================
        # 1. 禁止无意义特殊token
        # =================================================

        next_logits = block_invalid_special_tokens(
            next_logits
        )


        # =================================================
        # 2. Repetition penalty
        # =================================================

        next_logits = apply_repetition_penalty(

            logits=next_logits,

            generated_ids=generated_ids,

            penalty=REPETITION_PENALTY,

            window=REPETITION_WINDOW,

        )


        # =================================================
        # 3. No-repeat n-gram
        # =================================================

        banned_tokens = get_banned_ngram_tokens(

            generated_ids=generated_ids,

            ngram_size=NO_REPEAT_NGRAM_SIZE,

        )


        for token_id in banned_tokens:

            if (
                0
                <= token_id
                < next_logits.shape[-1]
            ):

                next_logits[
                    0,
                    token_id
                ] = -float(
                    "inf"
                )


        # =================================================
        # 安全检查
        # =================================================

        if not torch.isfinite(
            next_logits
        ).any():

            raise RuntimeError(
                "Anti-repetition约束后所有token logits均不可用"
            )


        # =================================================
        # 4. Penalized greedy decoding
        #
        # 不引入sampling随机性。
        # 普通SFT和RAG-SFT完全相同。
        # =================================================

        next_token = torch.argmax(

            next_logits,

            dim=-1,

            keepdim=True,

        )


        next_token_id = int(
            next_token.item()
        )


        # EOS自然停止。
        if (
            eos_id is not None
            and
            next_token_id == eos_id
        ):

            break


        input_ids = torch.cat(

            [
                input_ids,
                next_token
            ],

            dim=1,

        )


        generated_ids.append(
            next_token_id
        )


    answer = tokenizer.decode(

        generated_ids,

        skip_special_tokens=True,

    ).strip()


    return (
        answer,
        generated_ids
    )


# =====================================================
# 简单文本诊断
# =====================================================

def answer_stats(
    answer
):

    word_count = len(
        re.findall(
            r"\b[\w'\-]+\b",
            answer
        )
    )


    return {

        "chars":
            len(
                answer
            ),

        "words":
            word_count,

    }


# =====================================================
# 单个问题比较
# =====================================================

def compare_one_question(
    question,
    save_result=True,
):

    contexts = retrieve(
        question,
        top_k=RETRIEVE_TOP_K
    )


    if not contexts:

        print(
            "\n没有检索到Evidence"
        )

        return None


    (
        prompt_ids,
        prompt_text,
        prompt_truncated,

    ) = build_prompt_ids(

        question,
        contexts,
        ragsft_config,
        max_new_tokens=MAX_NEW_TOKENS,

    )


    # =================================================
    # 同一个Prompt分别喂给：
    #
    # 1. 普通SFT epoch3
    # 2. RAG-SFT best
    #
    # 控制变量比较。
    # =================================================

    sft_answer, sft_output_ids = (
        generate_from_model(

            sft_model,

            sft_config,

            prompt_ids,

        )
    )


    ragsft_answer, ragsft_output_ids = (
        generate_from_model(

            ragsft_model,

            ragsft_config,

            prompt_ids,

        )
    )


    print(
        "\n\n================================="
    )

    print(
        "QUESTION"
    )

    print(
        "================================="
    )

    print(
        question
    )


    print(
        "\n========== RETRIEVED EVIDENCE =========="
    )


    for rank, item in enumerate(
        contexts,
        start=1,
    ):

        print(
            f"\nRank {rank}"
        )

        print(
            "score:",
            f"{item['score']:.4f}"
        )

        print(
            "source:",
            item[
                "source"
            ]
        )

        print(
            "chunk:",
            item[
                "chunk_id"
            ]
        )

        print(
            "text:"
        )

        print(
            item[
                "text"
            ][
                :700
            ]
        )


    print(
        "\n========== PROMPT =========="
    )

    print(
        "prompt tokens:",
        len(
            prompt_ids
        )
    )

    print(
        "evidence truncated:",
        prompt_truncated
    )


    print(
        "\n========== NORMAL SFT =========="
    )

    print(
        sft_answer
    )


    print(
        "\n========== RAG-SFT =========="
    )

    print(
        ragsft_answer
    )


    sft_stats = answer_stats(
        sft_answer
    )

    ragsft_stats = answer_stats(
        ragsft_answer
    )


    print(
        "\n========== LENGTH =========="
    )

    print(
        "Normal SFT:",
        sft_stats
    )

    print(
        "RAG-SFT:",
        ragsft_stats
    )


    result = {

        "question":
            question,

        "retrieve_top_k":
            RETRIEVE_TOP_K,

        "decoding": {

            "strategy":
                DECODING_STRATEGY,

            "max_new_tokens":
                MAX_NEW_TOKENS,

            "repetition_penalty":
                REPETITION_PENALTY,

            "no_repeat_ngram_size":
                NO_REPEAT_NGRAM_SIZE,

            "repetition_window":
                REPETITION_WINDOW,

        },

        "prompt_token_count":
            len(
                prompt_ids
            ),

        "evidence_truncated":
            prompt_truncated,

        "contexts": contexts,

        "normal_sft": {

            "checkpoint":
                str(
                    SFT_MODEL_FILE
                ),

            "answer":
                sft_answer,

            "generated_token_count":
                len(
                    sft_output_ids
                ),

            "stats":
                sft_stats,

        },

        "rag_sft": {

            "checkpoint":
                str(
                    RAGSFT_MODEL_FILE
                ),

            "answer":
                ragsft_answer,

            "generated_token_count":
                len(
                    ragsft_output_ids
                ),

            "stats":
                ragsft_stats,

        },

    }


    if save_result:

        with open(

            OUTPUT_FILE,

            "a",

            encoding="utf-8",

        ) as f:

            f.write(

                json.dumps(
                    result,
                    ensure_ascii=False
                )

                +
                "\n"

            )


    return result


# =====================================================
# 主程序
# =====================================================

def main():

    # 每次运行重新生成比较结果，
    # 避免旧测试混在一起。
    if OUTPUT_FILE.exists():

        OUTPUT_FILE.unlink()


    print(
        "\n================================="
    )

    print(
        "RAG-SFT controlled comparison"
    )

    print(
        "================================="
    )

    print(
        "普通SFT:"
    )

    print(
        SFT_MODEL_FILE
    )

    print(
        "\nRAG-SFT:"
    )

    print(
        RAGSFT_MODEL_FILE
    )

    print(
        "\n测试结果保存:"
    )

    print(
        OUTPUT_FILE
    )


    print(
        "\n解码策略:"
    )

    print(
        DECODING_STRATEGY
    )

    print(
        "MAX_NEW_TOKENS:",
        MAX_NEW_TOKENS
    )

    print(
        "REPETITION_PENALTY:",
        REPETITION_PENALTY
    )

    print(
        "NO_REPEAT_NGRAM_SIZE:",
        NO_REPEAT_NGRAM_SIZE
    )

    print(
        "REPETITION_WINDOW:",
        REPETITION_WINDOW
    )


    # =================================================
    # 固定诊断问题
    # =================================================

    if RUN_FIXED_TESTS:

        print(
            "\n================================="
        )

        print(
            "开始固定诊断问题"
        )

        print(
            "================================="
        )


        for index, question in enumerate(

            TEST_QUESTIONS,

            start=1,

        ):

            print(
                f"\n\n######## TEST {index}/{len(TEST_QUESTIONS)} ########"
            )

            compare_one_question(
                question,
                save_result=True
            )


    # =================================================
    # 交互模式
    # =================================================

    if RUN_INTERACTIVE:

        print(
            "\n\n================================="
        )

        print(
            "进入交互比较模式"
        )

        print(
            "输入 exit / quit / q 退出"
        )

        print(
            "================================="
        )


        while True:

            question = input(
                "\nUser:\n> "
            ).strip()


            if question.lower() in {

                "exit",
                "quit",
                "q",

            }:

                break


            if not question:
                continue


            compare_one_question(
                question,
                save_result=True
            )


    print(
        "\n================================="
    )

    print(
        "22_test_rag_sft.py 完成"
    )

    print(
        "结果:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "================================="
    )


if __name__ == "__main__":

    main()
