import json
import re
import hashlib
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer


# =====================================================
# 路径
#
# 当前文件：
# 项目根目录/scripts/ragsft/20_clean_rag_sft.py
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]


RAW_FILE = (
    BASE_DIR
    / "data"
    / "sft"
    / "ragsft"
    / "cnt_rag_sft_raw.jsonl"
)


OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "sft"
    / "ragsft"
    / "cnt_rag_sft_dataset_clean.jsonl"
)


REJECTED_FILE = (
    BASE_DIR
    / "data"
    / "sft"
    / "ragsft"
    / "cnt_rag_sft_rejected.jsonl"
)


REPORT_FILE = (
    BASE_DIR
    / "data"
    / "sft"
    / "ragsft"
    / "cnt_rag_sft_clean_report.json"
)


PAPER_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "CNT_clean"
)


TOKENIZER_FILE = (
    BASE_DIR
    / "tokenizer"
    / "cnt_tokenizer.json"
)


# =====================================================
# 与19保持一致
# =====================================================

TRAIN_CUTOFF_YEAR = 2024

CHUNK_SIZE = 1000
OVERLAP = 200
MIN_CHUNK_CHARS = 100

MAX_EVIDENCE_CHUNKS = 2
MAX_EVIDENCE_CHARS = 2200

MIN_ANSWER_WORDS = 70
MAX_ANSWER_WORDS = 280

MIN_QUESTION_CHARS = 20
MAX_QUESTION_CHARS = 1000


# =====================================================
# 与后续21 Dataset一致的token预算检查
#
# 这里只做“能否训练”的预检查。
# Evidence过长时允许后续21只截Evidence；
# 但如果Question + Answer本身已经导致Evidence连16 token
# 都放不下，则20直接剔除。
# =====================================================

MAX_LENGTH = 1024
MIN_EVIDENCE_TOKENS = 16


INSTRUCTION_HEADER = (
    "### Instruction:\n"
    "Use the following retrieved evidence to answer "
    "the CNT question accurately."
)

EVIDENCE_HEADER = "\n\nEvidence:\n"
QUESTION_HEADER = "\n\nQuestion:\n"
RESPONSE_HEADER = "\n\n### Response:\n"


# =====================================================
# QA类型
# =====================================================

VALID_QA_TYPES = {
    "growth_mechanism",
    "catalyst_effect",
    "process_parameter",
    "carbon_source",
    "structure_control",
    "characterization",
    "structure_property",
    "optimization",
    "failure_limitation",
    "quantitative_condition",
    "comparison",
    "synthesis_strategy",
}


# =====================================================
# 来源依赖表达
# =====================================================

FORBIDDEN_PHRASES = [
    "according to the paper",
    "according to this paper",
    "this paper",
    "this study",
    "in this study",
    "the authors",
    "the researchers",
    "the article",
    "in this work",
    "the present work",
    "the work reports",
    "the paper reports",
    "according to the evidence",
    "the provided evidence",
    "the retrieved evidence",
    "this chunk",
    "the chunk",
    "chunk c",
]


# =====================================================
# 领域范围标记（只统计，不做硬过滤）
#
# 现在不再要求每个Question/Answer必须出现CNT关键词。
# 允许保留与CNT研究高度相邻的材料科学知识，例如：
# graphene、graphitic carbon、nanowire、ALD、界面热输运、
# 电化学、催化、聚合物复合、结构-性能关系等。
#
# 这里只把样本标记为：
#   core_cnt
#   adjacent_materials
#
# 不因为 adjacent_materials 而删除。
# =====================================================

CNT_DOMAIN_PATTERNS = [
    r"\bcarbon\s+nanotubes?\b",
    r"\bcarbon\s+nanotube\b",
    r"\bCNTs?\b",
    r"\bSWCNTs?\b",
    r"\bMWCNTs?\b",
    r"\bVACNTs?\b",
    r"\bsingle[-\s]?walled\s+carbon\s+nanotubes?\b",
    r"\bmulti[-\s]?walled\s+carbon\s+nanotubes?\b",
    r"\bvertically\s+aligned\s+carbon\s+nanotubes?\b",
    r"\bcarbon\s+nanotube\s+(?:array|arrays|forest|forests|carpet|carpets)\b",
    r"\bCNT\s+(?:array|arrays|forest|forests|carpet|carpets|growth|synthesis|catalyst|catalysts)\b",
]

CNT_DOMAIN_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in CNT_DOMAIN_PATTERNS),
    flags=re.IGNORECASE,
)


# =====================================================
# 数值支持检查
# =====================================================

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z])"
    r"[-+]?"
    r"(?:\d+(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z])"
)


# =====================================================
# 基础检查
# =====================================================

for path, name in [
    (RAW_FILE, "RAG-SFT raw数据"),
    (PAPER_DIR, "历史论文目录"),
    (TOKENIZER_FILE, "Tokenizer"),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到{name}: {path}"
        )


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================
# 文本处理：必须与19保持一致
# =====================================================

def clean_source_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00ad", "")

    return text.strip()


def remove_appended_pipeline_text(text):
    patterns = [
        r"(?im)^\s*=+\s*$\n"
        r"\s*STAGE\s+1\s+PHYSICAL\s+COMPUTATION\s+RESULTS\s*:",
        r"(?im)^\s*STAGE\s+1\s+PHYSICAL\s+COMPUTATION\s+RESULTS\s*:",
        r"(?im)^\s*STAGE\s+2\s+PHYSICAL\s+COMPUTATION\s+RESULTS\s*:",
        r"(?im)^\s*STAGE\s+\d+\s+.*RESULTS\s*:",
    ]

    earliest = None

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            if earliest is None or match.start() < earliest:
                earliest = match.start()

    if earliest is not None:
        text = text[:earliest]

    return text.strip()


def remove_reference_section(text):
    patterns = [
        r"(?im)^\s*■\s*REFERENCES\s*$",
        r"(?im)^\s*#+\s*REFERENCES\s*$",
        r"(?im)^\s*REFERENCES\s*$",
    ]

    earliest = None

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            # 与19保持一致，防止文章前部误判
            if match.start() < 3000:
                continue

            if earliest is None or match.start() < earliest:
                earliest = match.start()

    if earliest is not None:
        text = text[:earliest]

    return text.strip()


def read_article(path):
    try:
        text = Path(path).read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception:
        return ""

    text = clean_source_text(text)
    text = remove_appended_pipeline_text(text)
    text = remove_reference_section(text)

    # 19中MAX_CHARS = 300000
    if len(text) > 300000:
        text = text[:300000]

    return text


# =====================================================
# SHA1 / Temporal
# =====================================================

def sha1_text(text):
    return hashlib.sha1(
        str(text).encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()


def contains_future_year(filename):
    years = re.findall(
        r"(?<!\d)(20\d{2})(?!\d)",
        str(filename),
    )

    for year in years:
        try:
            year_int = int(year)
        except ValueError:
            continue

        if year_int > TRAIN_CUTOFF_YEAR:
            return True

    return False


# =====================================================
# Chunk重建：必须与19保持一致
# =====================================================

def split_into_chunks(text):
    chunks = []

    step = CHUNK_SIZE - OVERLAP

    if step <= 0:
        raise ValueError(
            "OVERLAP必须小于CHUNK_SIZE"
        )

    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < text_length:
        end = min(
            start + CHUNK_SIZE,
            text_length,
        )

        chunk_text = text[start:end].strip()

        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunk_id = f"C{chunk_index:04d}"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                    "text": chunk_text,
                    "chunk_sha1": sha1_text(chunk_text),
                }
            )

        chunk_index += 1

        if end >= text_length:
            break

        start += step

    return chunks


# =====================================================
# Paper map
# =====================================================

def build_paper_map():
    files = sorted(
        PAPER_DIR.glob("*.md")
    )

    exact_map = {}
    lower_map = {}

    for path in files:
        exact_map[path.name] = path

        key = path.name.casefold()

        if key not in lower_map:
            lower_map[key] = path
        else:
            # 大小写同名冲突时禁止fallback
            lower_map[key] = None

    return files, exact_map, lower_map


def find_source_paper(
    source_file,
    exact_map,
    lower_map,
):
    if source_file in exact_map:
        return exact_map[source_file]

    return lower_map.get(
        source_file.casefold()
    )


# =====================================================
# Word / Question normalize
# =====================================================

def word_count(text):
    return len(
        re.findall(
            r"\b[\w'\-]+\b",
            str(text),
        )
    )


def normalize_question(text):
    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


# =====================================================
# Domain scope
#
# 只用于统计，不用于删除样本。
# =====================================================

def classify_domain_scope(question, answer):
    combined = (
        str(question)
        + "\n"
        + str(answer)
    )

    if CNT_DOMAIN_REGEX.search(combined):
        return "core_cnt"

    return "adjacent_materials"


# =====================================================
# Numeric support
# =====================================================

def normalize_numeric_text(text):
    text = str(text)

    # 常见Unicode负号/破折号统一
    for symbol in ["−", "–", "—", "‐"]:
        text = text.replace(symbol, "-")

    # 1,000 -> 1000
    text = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        text,
    )

    return text


def extract_numeric_values(text):
    text = normalize_numeric_text(text)

    values = []

    for match in NUMBER_PATTERN.findall(text):
        try:
            values.append(
                round(
                    float(match),
                    8,
                )
            )
        except Exception:
            continue

    return values


def numeric_claims_supported(
    answer,
    evidence_text,
):
    answer_numbers = extract_numeric_values(
        answer
    )

    if not answer_numbers:
        return True

    evidence_numbers = extract_numeric_values(
        evidence_text
    )

    evidence_set = set(
        evidence_numbers
    )

    for number in answer_numbers:
        if number not in evidence_set:
            return False

    return True


# =====================================================
# Tokenizer / Token budget
# =====================================================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)

pad_id = tokenizer.token_to_id("[PAD]")
eos_id = tokenizer.token_to_id("[EOS]")

instruction_header_ids = tokenizer.encode(
    INSTRUCTION_HEADER
).ids

evidence_header_ids = tokenizer.encode(
    EVIDENCE_HEADER
).ids

question_header_ids = tokenizer.encode(
    QUESTION_HEADER
).ids

response_header_ids = tokenizer.encode(
    RESPONSE_HEADER
).ids


def token_budget_info(
    question,
    answer,
    evidence_text,
):
    question_ids = tokenizer.encode(
        question
    ).ids

    answer_ids = tokenizer.encode(
        answer
    ).ids

    evidence_ids = tokenizer.encode(
        evidence_text
    ).ids

    if eos_id is not None:
        answer_ids_with_eos = (
            answer_ids
            + [eos_id]
        )
    else:
        answer_ids_with_eos = list(
            answer_ids
        )

    # 因为21会 full_ids[:-1]，
    # full_ids最多允许MAX_LENGTH + 1
    full_token_budget = MAX_LENGTH + 1

    mandatory_tokens = (
        len(instruction_header_ids)
        + len(evidence_header_ids)
        + len(question_header_ids)
        + len(question_ids)
        + len(response_header_ids)
        + len(answer_ids_with_eos)
    )

    available_for_evidence = (
        full_token_budget
        - mandatory_tokens
    )

    usable_evidence_tokens = max(
        0,
        min(
            len(evidence_ids),
            available_for_evidence,
        ),
    )

    return {
        "question_token_count": len(question_ids),
        "answer_token_count": len(answer_ids_with_eos),
        "evidence_token_count": len(evidence_ids),
        "evidence_token_budget": max(
            0,
            available_for_evidence,
        ),
        "usable_evidence_tokens": usable_evidence_tokens,
        "evidence_requires_truncation": (
            len(evidence_ids)
            > max(0, available_for_evidence)
        ),
    }


# =====================================================
# Source cache
# =====================================================

SOURCE_CACHE = {}


def get_source_record(
    source_file,
    exact_map,
    lower_map,
):
    if source_file in SOURCE_CACHE:
        return SOURCE_CACHE[source_file]

    paper_path = find_source_paper(
        source_file,
        exact_map,
        lower_map,
    )

    if paper_path is None:
        SOURCE_CACHE[source_file] = None
        return None

    article_text = read_article(
        paper_path
    )

    if not article_text:
        SOURCE_CACHE[source_file] = None
        return None

    chunks = split_into_chunks(
        article_text
    )

    chunk_map = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    record = {
        "paper_path": paper_path,
        "article_text": article_text,
        "source_sha1": sha1_text(article_text),
        "chunks": chunks,
        "chunk_map": chunk_map,
    }

    SOURCE_CACHE[source_file] = record

    return record


# =====================================================
# Reject helper
# =====================================================

def reject_record(
    raw_item,
    reason,
    detail=None,
):
    record = {
        "reason": reason,
        "source_file": str(
            raw_item.get(
                "source_file",
                "",
            )
        ),
        "instruction": str(
            raw_item.get(
                "instruction",
                "",
            )
        ),
    }

    if detail is not None:
        record["detail"] = str(detail)

    return record


# =====================================================
# 单条清洗
# =====================================================

def clean_one_item(
    item,
    exact_map,
    lower_map,
    seen_questions,
):
    if not isinstance(item, dict):
        return None, "item_not_dict", None

    # -------------------------------------------------
    # 必须字段
    # -------------------------------------------------

    required_fields = [
        "instruction",
        "output",
        "qa_type",
        "source_file",
        "source_sha1",
        "evidence_chunk_ids",
        "evidence_chunks",
        "evidence_text",
    ]

    missing = [
        field
        for field in required_fields
        if field not in item
    ]

    if missing:
        return (
            None,
            "missing_fields",
            ",".join(missing),
        )

    # -------------------------------------------------
    # 文本
    # -------------------------------------------------

    question = str(
        item.get(
            "instruction",
            "",
        )
    ).strip()

    answer = str(
        item.get(
            "output",
            "",
        )
    ).strip()

    qa_type = str(
        item.get(
            "qa_type",
            "",
        )
    ).strip()

    source_file = str(
        item.get(
            "source_file",
            "",
        )
    ).strip()

    source_sha1 = str(
        item.get(
            "source_sha1",
            "",
        )
    ).strip()

    if not question:
        return None, "empty_question", None

    if not answer:
        return None, "empty_answer", None

    if not source_file:
        return None, "empty_source_file", None

    if (
        len(question)
        < MIN_QUESTION_CHARS
    ):
        return None, "question_too_short", None

    if (
        len(question)
        > MAX_QUESTION_CHARS
    ):
        return None, "question_too_long", None

    answer_words = word_count(
        answer
    )

    if answer_words < MIN_ANSWER_WORDS:
        return None, "answer_too_short", answer_words

    if answer_words > MAX_ANSWER_WORDS:
        return None, "answer_too_long", answer_words

    if qa_type not in VALID_QA_TYPES:
        return None, "invalid_qa_type", qa_type

    # -------------------------------------------------
    # Markdown / source-dependent language
    # -------------------------------------------------

    if "```" in question or "```" in answer:
        return None, "markdown_fence", None

    combined_lower = (
        question
        + "\n"
        + answer
    ).lower()

    for phrase in FORBIDDEN_PHRASES:
        if phrase in combined_lower:
            return (
                None,
                "source_dependent_language",
                phrase,
            )

    # -------------------------------------------------
    # Temporal
    # -------------------------------------------------

    if contains_future_year(
        source_file
    ):
        return None, "future_year_source", source_file

    # -------------------------------------------------
    # Source重建
    # -------------------------------------------------

    source_record = get_source_record(
        source_file,
        exact_map,
        lower_map,
    )

    if source_record is None:
        return None, "source_file_not_found_or_empty", source_file

    if source_sha1 != source_record["source_sha1"]:
        return (
            None,
            "source_sha1_mismatch",
            None,
        )

    chunk_map = source_record[
        "chunk_map"
    ]

    # -------------------------------------------------
    # Chunk配置一致性
    # -------------------------------------------------

    raw_chunk_size = item.get(
        "chunk_size",
        CHUNK_SIZE,
    )

    raw_chunk_overlap = item.get(
        "chunk_overlap",
        OVERLAP,
    )

    try:
        raw_chunk_size = int(
            raw_chunk_size
        )

        raw_chunk_overlap = int(
            raw_chunk_overlap
        )
    except Exception:
        return None, "invalid_chunk_config", None

    if raw_chunk_size != CHUNK_SIZE:
        return (
            None,
            "chunk_size_mismatch",
            raw_chunk_size,
        )

    if raw_chunk_overlap != OVERLAP:
        return (
            None,
            "chunk_overlap_mismatch",
            raw_chunk_overlap,
        )

    # -------------------------------------------------
    # Evidence IDs
    # -------------------------------------------------

    raw_ids = item.get(
        "evidence_chunk_ids",
        [],
    )

    if not isinstance(raw_ids, list):
        return None, "evidence_ids_not_list", None

    selected_ids = []

    for raw_id in raw_ids:
        chunk_id = str(
            raw_id
        ).strip()

        if not chunk_id:
            continue

        if chunk_id not in selected_ids:
            selected_ids.append(
                chunk_id
            )

    if not selected_ids:
        return None, "no_evidence_ids", None

    if len(selected_ids) > MAX_EVIDENCE_CHUNKS:
        return (
            None,
            "too_many_evidence_chunks",
            len(selected_ids),
        )

    for chunk_id in selected_ids:
        if chunk_id not in chunk_map:
            return (
                None,
                "chunk_id_not_found",
                chunk_id,
            )

    # -------------------------------------------------
    # Evidence chunks结构验证
    # -------------------------------------------------

    raw_chunks = item.get(
        "evidence_chunks",
        [],
    )

    if not isinstance(raw_chunks, list):
        return None, "evidence_chunks_not_list", None

    raw_chunk_dict = {}

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            return None, "invalid_evidence_chunk_item", None

        chunk_id = str(
            raw_chunk.get(
                "chunk_id",
                "",
            )
        ).strip()

        if not chunk_id:
            return None, "evidence_chunk_missing_id", None

        if chunk_id in raw_chunk_dict:
            return (
                None,
                "duplicate_evidence_chunk_id",
                chunk_id,
            )

        raw_chunk_dict[chunk_id] = raw_chunk

    if set(raw_chunk_dict.keys()) != set(selected_ids):
        return (
            None,
            "evidence_chunks_ids_mismatch",
            None,
        )

    # -------------------------------------------------
    # 从源论文重建canonical evidence
    # -------------------------------------------------

    canonical_chunks = []

    for chunk_id in selected_ids:
        canonical = chunk_map[
            chunk_id
        ]

        raw_chunk = raw_chunk_dict[
            chunk_id
        ]

        # start/end
        try:
            raw_start = int(
                raw_chunk.get(
                    "start_char"
                )
            )

            raw_end = int(
                raw_chunk.get(
                    "end_char"
                )
            )
        except Exception:
            return (
                None,
                "invalid_chunk_position",
                chunk_id,
            )

        if raw_start != canonical["start_char"]:
            return (
                None,
                "chunk_start_mismatch",
                chunk_id,
            )

        if raw_end != canonical["end_char"]:
            return (
                None,
                "chunk_end_mismatch",
                chunk_id,
            )

        raw_text = str(
            raw_chunk.get(
                "text",
                "",
            )
        )

        if raw_text != canonical["text"]:
            return (
                None,
                "chunk_text_mismatch",
                chunk_id,
            )

        raw_sha1 = str(
            raw_chunk.get(
                "chunk_sha1",
                "",
            )
        ).strip()

        if raw_sha1 != canonical["chunk_sha1"]:
            return (
                None,
                "chunk_sha1_mismatch",
                chunk_id,
            )

        # 二次本地重算SHA1
        if sha1_text(raw_text) != raw_sha1:
            return (
                None,
                "chunk_sha1_recompute_failed",
                chunk_id,
            )

        canonical_chunks.append(
            {
                "chunk_id": canonical["chunk_id"],
                "start_char": canonical["start_char"],
                "end_char": canonical["end_char"],
                "chunk_sha1": canonical["chunk_sha1"],
                "text": canonical["text"],
            }
        )

    # -------------------------------------------------
    # Evidence text重建
    # -------------------------------------------------

    canonical_evidence_text = "\n\n".join(
        (
            f"[{chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )
        for chunk in canonical_chunks
    )

    raw_evidence_text = str(
        item.get(
            "evidence_text",
            "",
        )
    ).strip()

    if raw_evidence_text != canonical_evidence_text:
        return (
            None,
            "evidence_text_mismatch",
            None,
        )

    evidence_char_count = sum(
        len(chunk["text"])
        for chunk in canonical_chunks
    )

    if evidence_char_count > MAX_EVIDENCE_CHARS:
        return (
            None,
            "evidence_too_long",
            evidence_char_count,
        )

    # -------------------------------------------------
    # 数值支持
    # -------------------------------------------------

    if not numeric_claims_supported(
        answer,
        canonical_evidence_text,
    ):
        return (
            None,
            "unsupported_numeric_claim",
            None,
        )

    # -------------------------------------------------
    # 全局Question去重
    # -------------------------------------------------

    question_key = normalize_question(
        question
    )

    if question_key in seen_questions:
        return None, "duplicate_question", None

    # -------------------------------------------------
    # Token budget
    # -------------------------------------------------

    budget = token_budget_info(
        question,
        answer,
        canonical_evidence_text,
    )

    if (
        budget["evidence_token_budget"]
        < MIN_EVIDENCE_TOKENS
    ):
        return (
            None,
            "insufficient_evidence_token_budget",
            budget["evidence_token_budget"],
        )

    # -------------------------------------------------
    # 通过
    # -------------------------------------------------

    seen_questions.add(
        question_key
    )

    # -------------------------------------------------
    # 领域范围只做标记，不做删除
    # -------------------------------------------------

    domain_scope = classify_domain_scope(
        question,
        answer,
    )

    clean_item = {
        "instruction": question,
        "input": "",
        "output": answer,
        "qa_type": qa_type,
        "domain_scope": domain_scope,

        "source_file": source_file,
        "source_sha1": source_record["source_sha1"],

        "evidence_chunk_ids": selected_ids,
        "evidence_chunks": canonical_chunks,
        "evidence_text": canonical_evidence_text,

        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": OVERLAP,

        "evidence_char_count": evidence_char_count,
        "answer_word_count": answer_words,

        "question_token_count": budget["question_token_count"],
        "answer_token_count": budget["answer_token_count"],
        "evidence_token_count": budget["evidence_token_count"],
        "evidence_token_budget": budget["evidence_token_budget"],
        "evidence_requires_truncation": budget[
            "evidence_requires_truncation"
        ],

        "generator_model": item.get(
            "generator_model",
            "",
        ),
        "thinking_enabled": bool(
            item.get(
                "thinking_enabled",
                False,
            )
        ),
        "schema_version": item.get(
            "schema_version",
            "rag_sft_grounded_chunk_ids",
        ),
    }

    return clean_item, None, None


# =====================================================
# Percentile helper
# =====================================================

def percentile(values, p):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * p
        / 100.0
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    fraction = position - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def summary_stats(values):
    if not values:
        return {
            "count": 0,
        }

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


# =====================================================
# main
# =====================================================

def main():
    print(
        "================================="
    )
    print(
        "YanlongGPT-CNT RAG-SFT Cleaner"
    )
    print(
        "================================="
    )

    print(
        "输入:"
    )
    print(
        RAW_FILE
    )

    print(
        "\n输出:"
    )
    print(
        OUTPUT_FILE
    )

    print(
        "\n拒绝记录:"
    )
    print(
        REJECTED_FILE
    )

    print(
        "\nTokenizer:"
    )
    print(
        TOKENIZER_FILE
    )

    paper_files, exact_map, lower_map = (
        build_paper_map()
    )

    print(
        "\n历史论文数量:",
        len(paper_files),
    )

    print(
        "领域策略: 保留CNT核心 + 相邻材料科学；"
        "只标记domain_scope，不做CNT-only硬过滤"
    )

    seen_questions = set()

    rejection_counter = Counter()
    qa_type_counter = Counter()
    domain_scope_counter = Counter()
    evidence_count_counter = Counter()

    answer_word_lengths = []
    question_token_lengths = []
    answer_token_lengths = []
    evidence_token_lengths = []
    evidence_token_budgets = []

    raw_count = 0
    clean_count = 0
    json_failed = 0
    evidence_truncate_count = 0

    output_tmp = OUTPUT_FILE.with_suffix(
        OUTPUT_FILE.suffix + ".tmp"
    )

    rejected_tmp = REJECTED_FILE.with_suffix(
        REJECTED_FILE.suffix + ".tmp"
    )

    with open(
        RAW_FILE,
        "r",
        encoding="utf-8",
    ) as fin, open(
        output_tmp,
        "w",
        encoding="utf-8",
    ) as fout, open(
        rejected_tmp,
        "w",
        encoding="utf-8",
    ) as freject:

        for line_number, line in enumerate(
            fin,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            raw_count += 1

            try:
                item = json.loads(
                    line
                )
            except Exception as e:
                json_failed += 1
                rejection_counter[
                    "json_parse_failed"
                ] += 1

                freject.write(
                    json.dumps(
                        {
                            "reason": "json_parse_failed",
                            "line_number": line_number,
                            "detail": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                continue

            clean_item, reason, detail = (
                clean_one_item(
                    item=item,
                    exact_map=exact_map,
                    lower_map=lower_map,
                    seen_questions=seen_questions,
                )
            )

            if clean_item is None:
                rejection_counter[
                    reason
                ] += 1

                rejected = reject_record(
                    raw_item=item,
                    reason=reason,
                    detail=detail,
                )

                rejected[
                    "line_number"
                ] = line_number

                freject.write(
                    json.dumps(
                        rejected,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                continue

            fout.write(
                json.dumps(
                    clean_item,
                    ensure_ascii=False,
                )
                + "\n"
            )

            clean_count += 1

            qa_type_counter[
                clean_item["qa_type"]
            ] += 1

            domain_scope_counter[
                clean_item["domain_scope"]
            ] += 1

            evidence_count_counter[
                len(
                    clean_item[
                        "evidence_chunk_ids"
                    ]
                )
            ] += 1

            answer_word_lengths.append(
                clean_item[
                    "answer_word_count"
                ]
            )

            question_token_lengths.append(
                clean_item[
                    "question_token_count"
                ]
            )

            answer_token_lengths.append(
                clean_item[
                    "answer_token_count"
                ]
            )

            evidence_token_lengths.append(
                clean_item[
                    "evidence_token_count"
                ]
            )

            evidence_token_budgets.append(
                clean_item[
                    "evidence_token_budget"
                ]
            )

            if clean_item[
                "evidence_requires_truncation"
            ]:
                evidence_truncate_count += 1

    # 原子替换，避免运行中断留下半个clean文件
    output_tmp.replace(
        OUTPUT_FILE
    )

    rejected_tmp.replace(
        REJECTED_FILE
    )

    report = {
        "raw_file": str(RAW_FILE),
        "output_file": str(OUTPUT_FILE),
        "rejected_file": str(REJECTED_FILE),

        "raw_count": raw_count,
        "json_failed": json_failed,
        "clean_count": clean_count,
        "rejected_count": raw_count - clean_count,
        "retention_rate": (
            clean_count / raw_count
            if raw_count > 0
            else 0.0
        ),

        "rejection_reasons": dict(
            rejection_counter
        ),

        "qa_type_distribution": dict(
            qa_type_counter
        ),

        "domain_scope_distribution": dict(
            domain_scope_counter
        ),

        "evidence_chunk_count_distribution": {
            str(key): value
            for key, value in
            evidence_count_counter.items()
        },

        "evidence_requires_truncation_count": (
            evidence_truncate_count
        ),

        "answer_word_stats": summary_stats(
            answer_word_lengths
        ),

        "question_token_stats": summary_stats(
            question_token_lengths
        ),

        "answer_token_stats": summary_stats(
            answer_token_lengths
        ),

        "evidence_token_stats": summary_stats(
            evidence_token_lengths
        ),

        "evidence_token_budget_stats": summary_stats(
            evidence_token_budgets
        ),

        "config": {
            "train_cutoff_year": TRAIN_CUTOFF_YEAR,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": OVERLAP,
            "max_evidence_chunks": MAX_EVIDENCE_CHUNKS,
            "max_evidence_chars": MAX_EVIDENCE_CHARS,
            "min_answer_words": MIN_ANSWER_WORDS,
            "max_answer_words": MAX_ANSWER_WORDS,
            "max_length": MAX_LENGTH,
            "min_evidence_tokens": MIN_EVIDENCE_TOKENS,
            "domain_policy": "annotate_only_no_hard_filter",
        },
    }

    REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n================================="
    )
    print(
        "RAG-SFT清洗完成"
    )
    print(
        "================================="
    )

    print(
        "原始样本:",
        raw_count,
    )

    print(
        "JSON失败:",
        json_failed,
    )

    print(
        "通过样本:",
        clean_count,
    )

    print(
        "拒绝样本:",
        raw_count - clean_count,
    )

    if raw_count > 0:
        print(
            "保留率:",
            f"{clean_count / raw_count * 100:.2f}%",
        )

    print(
        "\n拒绝原因:"
    )

    if rejection_counter:
        for reason, count in (
            rejection_counter.most_common()
        ):
            print(
                f"- {reason}: {count}"
            )
    else:
        print(
            "- 无"
        )

    print(
        "\nQA类型分布:"
    )

    for qa_type, count in sorted(
        qa_type_counter.items()
    ):
        print(
            f"- {qa_type}: {count}"
        )

    print(
        "\n领域范围分布（只统计，不过滤）:"
    )

    for domain_scope, count in sorted(
        domain_scope_counter.items()
    ):
        print(
            f"- {domain_scope}: {count}"
        )

    print(
        "\nEvidence chunk数量:"
    )

    for chunk_count, count in sorted(
        evidence_count_counter.items()
    ):
        print(
            f"- {chunk_count} chunk: {count}"
        )

    print(
        "\n后续21需要截断Evidence的样本:",
        evidence_truncate_count,
    )

    def print_stats(name, values):
        stats = summary_stats(values)

        if stats.get("count", 0) == 0:
            print(
                f"{name}: 无数据"
            )
            return

        print(
            f"{name}: "
            f"mean={stats['mean']:.1f}, "
            f"p50={stats['p50']:.1f}, "
            f"p95={stats['p95']:.1f}, "
            f"max={stats['max']}"
        )

    print(
        "\n长度统计:"
    )

    print_stats(
        "Answer words",
        answer_word_lengths,
    )

    print_stats(
        "Question tokens",
        question_token_lengths,
    )

    print_stats(
        "Answer tokens",
        answer_token_lengths,
    )

    print_stats(
        "Evidence tokens",
        evidence_token_lengths,
    )

    print_stats(
        "Evidence token budget",
        evidence_token_budgets,
    )

    print(
        "\nClean数据:"
    )
    print(
        OUTPUT_FILE
    )

    print(
        "\nRejected数据:"
    )
    print(
        REJECTED_FILE
    )

    print(
        "\nReport:"
    )
    print(
        REPORT_FILE
    )

    print(
        "\n下一步:"
    )
    print(
        "21_rag_sft_dataset.py"
    )

    print(
        "================================="
    )


if __name__ == "__main__":
    main()
