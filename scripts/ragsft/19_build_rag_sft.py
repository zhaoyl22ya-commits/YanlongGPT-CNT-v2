import json
import re
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


# =====================================================
# 路径
#
# 当前文件：
# project/scripts/ragsft/19_build_rag_sft.py
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]


# =====================================================
# API
# =====================================================

KEY_FILE = (
    BASE_DIR
    / "scripts"
    / "key"
)

# 并行智算云中的模型ID
MODEL = "DeepSeek-V4.1-Flash"

# 继续使用并行智算云，不改成DeepSeek官方地址
BASE_URL = "https://llmapi.paratera.com/v1/"


def load_api_key():

    if not KEY_FILE.exists():

        raise FileNotFoundError(
            f"找不到API key文件: {KEY_FILE}"
        )

    text = KEY_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    api_key = None

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        if line.upper().startswith("API_KEY"):

            if ":" in line:

                api_key = line.split(
                    ":",
                    1
                )[1].strip()

            elif "=" in line:

                api_key = line.split(
                    "=",
                    1
                )[1].strip()

        elif line.startswith("sk-"):

            api_key = line

    if api_key is None:

        raise ValueError(
            "key文件中没有找到API_KEY"
        )

    return api_key


API_KEY = load_api_key()


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=300.0
)


# =====================================================
# 输入历史论文
#
# 必须保持 <= 2024
# =====================================================

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "CNT_clean"
)


# =====================================================
# 输出目录
# =====================================================

RAGSFT_DIR = (
    BASE_DIR
    / "data"
    / "sft"
    / "ragsft"
)


OUTPUT_FILE = (
    RAGSFT_DIR
    / "cnt_rag_sft_raw.jsonl"
)


PROCESSED_FILE = (
    RAGSFT_DIR
    / "processed_files.txt"
)


FAILED_DIR = (
    RAGSFT_DIR
    / "failed"
)


RAW_RESPONSE_DIR = (
    RAGSFT_DIR
    / "raw_failed_responses"
)


SKIPPED_FUTURE_FILE = (
    RAGSFT_DIR
    / "skipped_future_files.txt"
)


RAGSFT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FAILED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RAW_RESPONSE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# Temporal split
# =====================================================

TRAIN_CUTOFF_YEAR = 2024


# =====================================================
# Chunk参数
#
# 与当前正式RAG保持：
# 1000 chars
# 200 overlap
# =====================================================

CHUNK_SIZE = 1000

OVERLAP = 200

MIN_CHUNK_CHARS = 100


# =====================================================
# API参数
# =====================================================

# TEST阶段20篇时48已经完全够用。
#
# 20篇确认稳定以后，全量时可以改：
#
# MAX_WORKERS = 96
#
MAX_WORKERS = 96


MAX_CHARS = 300000


# thinking开启后保留较充裕输出空间
MAX_OUTPUT_TOKENS = 30000


# 每篇最多尝试2次。
#
# Flash本身便宜，但没必要因为一篇异常文章
# 连续付3次完整长文本输入费用。
MAX_RETRIES = 2


# =====================================================
# Thinking
# =====================================================

ENABLE_THINKING = True


# =====================================================
# TEST
#
# 当前先跑20篇。
#
# 检查完成后：
#
# TEST_LIMIT = None
# =====================================================

TEST_LIMIT = None


# =====================================================
# QA数量
# =====================================================

MAX_QA_PER_PAPER = 4


# =====================================================
# Answer长度
#
# 不强迫回答写长。
#
# 简单问题：
# 80-160 words
#
# 复杂问题：
# 120-220 / 160-260
#
# 程序最终允许：
# 70-280
# =====================================================

MIN_ANSWER_WORDS = 70

MAX_ANSWER_WORDS = 280


# =====================================================
# Evidence约束
#
# 最多2个chunk。
#
# 2 x 1000 chars + Question + Answer
# 后续更适合1024-token小模型。
# =====================================================

MAX_EVIDENCE_CHUNKS = 2

MAX_EVIDENCE_CHARS = 2200


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

    "synthesis_strategy"

}


# =====================================================
# 禁止来源依赖表达
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

    "chunk c"

]


# =====================================================
# Prompt
# =====================================================

SYSTEM_PROMPT = r"""
You are an expert in carbon nanotube (CNT) synthesis,
chemical vapor deposition, catalyst chemistry,
nanomaterial characterization, process optimization,
and scientific evidence-grounded reasoning.

Your task is to create HIGH-QUALITY RAG-aware
supervised fine-tuning data for a domain-specific
CNT scientific assistant.

You will receive ONE scientific article divided into
traceable source chunks.

Each source chunk has an immutable identifier:

C0000
C0001
C0002
...

Your task is to reason over the article, locate
scientifically valuable evidence, and construct
evidence-grounded question-answer pairs.


=========================================================
1. REASONING ORDER
=========================================================

For every candidate QA pair, reason internally in this order:

1. Identify a scientifically valuable CNT fact,
   mechanism, process relationship, experimental condition,
   optimization rule, characterization result, or limitation.

2. Locate the exact source chunk or chunks that directly
   support that knowledge.

3. Formulate a standalone scientific question.

4. Write an expert answer using ONLY information that can
   be supported by the selected chunks.

Do NOT expose your reasoning process.

Return only the requested JSON.


=========================================================
2. EVIDENCE SELECTION
=========================================================

Evidence quality is critical.

Normally select ONE chunk.

Select TWO chunks only when both are required to support
the complete answer.

Do not select extra chunks merely because they discuss
a similar topic.

If the desired answer would require information scattered
across more than two chunks, simplify the question so it
can be answered from one or two chunks.

Every numerical value, mechanism, comparison,
experimental parameter and scientific conclusion in the
answer must be supported by the selected chunks.


=========================================================
3. EVIDENCE TRACEABILITY
=========================================================

For every QA pair return only the exact chunk IDs that
support the answer.

Example:

"evidence_chunk_ids": ["C0014"]

or:

"evidence_chunk_ids": ["C0014", "C0015"]

Do NOT copy the source text into the JSON.

Do NOT create quotations.

Do NOT invent chunk IDs.

The Python pipeline will retrieve the original text
directly from the source article using these IDs.

Therefore selecting the correct chunk IDs is extremely
important.


=========================================================
4. PREFER STRONG DIRECT EVIDENCE
=========================================================

Prefer detailed passages from:

- Experimental Procedures / Methods
- Results
- Discussion
- mechanism analysis
- quantitative comparisons
- characterization results

Prefer detailed experimental or mechanistic passages over
generic sentences in the abstract when better evidence
exists elsewhere.

Do not use:

- bibliography entries
- reference-list text
- article metadata
- author information

as scientific evidence.


=========================================================
5. SCIENTIFIC ACCURACY
=========================================================

Never invent:

- temperatures
- pressures
- gas flow rates
- catalyst compositions
- catalyst particle sizes
- CNT diameters
- CNT lengths
- growth rates
- Raman peak positions
- ID/IG values
- yields
- electrical properties
- mechanisms
- performance values

Every factual sentence in the answer must be supported by
the selected evidence chunks.

If a fact cannot be supported by one or two chunks,
do not include it.


=========================================================
6. QUESTION VALUE
=========================================================

Prefer questions that a CNT researcher might genuinely
ask during:

- experimental design
- catalyst selection
- process optimization
- mechanism analysis
- troubleshooting
- characterization interpretation
- morphology control
- synthesis comparison

High-value questions include:

- Why does a growth parameter affect CNT morphology?
- How does catalyst composition influence growth?
- Which quantitative conditions produce a target structure?
- Why does a synthesis route fail under certain conditions?
- What does a Raman/TEM/SEM result imply?
- How does a structural change affect performance?
- What trade-off controls process optimization?

Avoid trivial encyclopedia questions such as:

"What is CNT?"
"What is CVD?"
"What are CNTs used for?"

Avoid questions about journal names, author names,
publication years or article metadata.


=========================================================
7. QUESTION INDEPENDENCE
=========================================================

Every question must be understandable without access
to the original article.

BAD:

"What temperature was used in this study?"

GOOD:

"What growth-temperature range promotes vertically aligned
CNT formation in Fe-catalyzed thermal CVD under the
specified hydrocarbon atmosphere?"

Include enough scientific context in the question to make
it independently meaningful.


=========================================================
8. QUESTION DIVERSITY
=========================================================

Generate normally 2-4 high-value QA pairs per article.

Quality is more important than quantity.

Questions from the same article should cover different
scientific aspects whenever possible.

Do not generate multiple questions that test essentially
the same fact.

Returning fewer questions is acceptable.

Returning zero questions is allowed when the article does
not contain sufficiently useful CNT knowledge.


=========================================================
9. ANSWER LENGTH
=========================================================

Use adaptive answer length.

Simple quantitative or characterization questions:
approximately 80-160 English words.

Mechanism, comparison or structure-property questions:
approximately 120-220 English words.

Complex synthesis or multivariable questions:
approximately 160-260 English words.

Absolute maximum:
280 English words.

Do NOT lengthen an answer merely to reach a target.

Every sentence must contain useful, evidence-supported
scientific information.


=========================================================
10. ANSWER STYLE
=========================================================

Answer the question directly.

Write the answer as scientific domain knowledge.

Do not mention:

- paper
- article
- authors
- study
- evidence
- chunk
- retrieved context

BAD:

"According to the paper, the authors found that..."

GOOD:

"Increasing catalyst particle size can broaden the CNT
diameter distribution because..."


=========================================================
11. QA TYPE
=========================================================

Each pair must use exactly one of:

growth_mechanism
catalyst_effect
process_parameter
carbon_source
structure_control
characterization
structure_property
optimization
failure_limitation
quantitative_condition
comparison
synthesis_strategy


=========================================================
12. JSON OUTPUT
=========================================================

Return ONLY one valid JSON object.

No Markdown.

No explanation outside the JSON.

Use exactly this structure:

{
  "data": [
    {
      "instruction": "standalone scientific question",
      "input": "",
      "output": "grounded scientific answer",
      "qa_type": "one allowed qa_type",
      "evidence_chunk_ids": ["C0014"]
    }
  ]
}

For two supporting chunks:

{
  "data": [
    {
      "instruction": "...",
      "input": "",
      "output": "...",
      "qa_type": "...",
      "evidence_chunk_ids": ["C0014", "C0015"]
    }
  ]
}

Never return more than two evidence chunk IDs per QA pair.
"""


# =====================================================
# 文本清理
# =====================================================

def clean_source_text(text):

    if not isinstance(
        text,
        str
    ):

        return ""

    text = text.replace(
        "\x00",
        ""
    )

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.replace(
        "\u00ad",
        ""
    )

    return text.strip()


# =====================================================
# 删除历史pipeline附加内容
# =====================================================

def remove_appended_pipeline_text(
    text
):

    patterns = [

        r"(?im)^\s*=+\s*$\n"
        r"\s*STAGE\s+1\s+PHYSICAL\s+COMPUTATION\s+RESULTS\s*:",

        r"(?im)^\s*STAGE\s+1\s+PHYSICAL\s+COMPUTATION\s+RESULTS\s*:",

        r"(?im)^\s*STAGE\s+2\s+PHYSICAL\s+COMPUTATION\s+RESULTS\s*:",

        r"(?im)^\s*STAGE\s+\d+\s+.*RESULTS\s*:"

    ]

    earliest = None

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            if (
                earliest is None
                or
                match.start() < earliest
            ):

                earliest = match.start()

    if earliest is not None:

        text = text[
            :earliest
        ]

    return text.strip()


# =====================================================
# 删除References
# =====================================================

def remove_reference_section(
    text
):

    patterns = [

        r"(?im)^\s*■\s*REFERENCES\s*$",

        r"(?im)^\s*#+\s*REFERENCES\s*$",

        r"(?im)^\s*REFERENCES\s*$"

    ]

    earliest = None

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:

            # 防止误伤文章前部
            if match.start() < 3000:

                continue

            if (
                earliest is None
                or
                match.start() < earliest
            ):

                earliest = match.start()

    if earliest is not None:

        text = text[
            :earliest
        ]

    return text.strip()


# =====================================================
# 读取论文
# =====================================================

def read_article(
    path
):

    try:

        text = Path(path).read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""

    text = clean_source_text(
        text
    )

    text = remove_appended_pipeline_text(
        text
    )

    text = remove_reference_section(
        text
    )

    if len(text) > MAX_CHARS:

        text = text[
            :MAX_CHARS
        ]

    return text


# =====================================================
# Temporal检查
# =====================================================

def contains_future_year(
    filename
):

    years = re.findall(

        r"(?<!\d)(20\d{2})(?!\d)",

        filename

    )

    for year in years:

        try:

            year_int = int(
                year
            )

        except ValueError:

            continue

        if year_int > TRAIN_CUTOFF_YEAR:

            return True

    return False


# =====================================================
# SHA1
# =====================================================

def sha1_text(
    text
):

    return hashlib.sha1(

        text.encode(
            "utf-8",
            errors="ignore"
        )

    ).hexdigest()


# =====================================================
# 切Chunk
# =====================================================

def split_into_chunks(
    text
):

    chunks = []

    step = (
        CHUNK_SIZE
        -
        OVERLAP
    )

    if step <= 0:

        raise ValueError(
            "OVERLAP必须小于CHUNK_SIZE"
        )

    start = 0

    chunk_index = 0

    text_length = len(
        text
    )

    while start < text_length:

        end = min(

            start
            +
            CHUNK_SIZE,

            text_length

        )

        chunk_text = text[
            start:end
        ].strip()

        if len(
            chunk_text
        ) >= MIN_CHUNK_CHARS:

            chunk_id = (
                f"C{chunk_index:04d}"
            )

            chunks.append(
                {
                    "chunk_id":
                        chunk_id,

                    "start_char":
                        start,

                    "end_char":
                        end,

                    "text":
                        chunk_text,

                    "chunk_sha1":
                        sha1_text(
                            chunk_text
                        )
                }
            )

        chunk_index += 1

        if end >= text_length:

            break

        start += step

    return chunks


# =====================================================
# 给API看的Chunk格式
# =====================================================

def format_chunks_for_api(
    chunks
):

    blocks = []

    for chunk in chunks:

        block = (

            f"\n"
            f"===== {chunk['chunk_id']} =====\n"
            f"{chunk['text']}\n"
            f"===== END {chunk['chunk_id']} =====\n"

        )

        blocks.append(
            block
        )

    return "\n".join(
        blocks
    )


# =====================================================
# JSON解析
# =====================================================

def parse_json_response(
    text
):

    if text is None:

        raise ValueError(
            "模型content为空"
        )

    text = str(
        text
    ).strip()

    if not text:

        raise ValueError(
            "模型content为空字符串"
        )

    # JSON模式理论上不需要，
    # 这里继续保留容错。
    if text.startswith(
        "```"
    ):

        text = re.sub(

            r"^```(?:json)?\s*",

            "",

            text,

            flags=re.IGNORECASE

        )

        text = re.sub(

            r"\s*```$",

            "",

            text

        )

        text = text.strip()

    try:

        return json.loads(
            text
        )

    except Exception:

        pass

    # 二次容错
    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start != -1
        and
        end != -1
        and
        end > start
    ):

        candidate = text[
            start:
            end + 1
        ]

        try:

            return json.loads(
                candidate
            )

        except Exception:

            pass

    raise ValueError(
        "无法解析模型返回JSON"
    )


# =====================================================
# Word count
# =====================================================

def word_count(
    text
):

    return len(

        re.findall(

            r"\b[\w'\-]+\b",

            str(text)

        )

    )


# =====================================================
# Question标准化
# =====================================================

def normalize_question(
    text
):

    return " ".join(

        str(text)
        .lower()
        .strip()
        .split()

    )


# =====================================================
# 提取数值
#
# 用于做Evidence硬验证。
# =====================================================

NUMBER_PATTERN = re.compile(

    r"(?<![A-Za-z])"
    r"[-+]?"
    r"(?:\d+\.\d+|\d+)"
    r"(?:[eE][-+]?\d+)?"
    r"(?![A-Za-z])"

)


def extract_numeric_values(
    text
):

    values = []

    for match in NUMBER_PATTERN.findall(
        str(text)
    ):

        try:

            values.append(

                round(
                    float(match),
                    8
                )

            )

        except Exception:

            continue

    return values


def numeric_claims_supported(
    answer,
    evidence_text
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
# 验证并回填Evidence
#
# 关键：
#
# LLM只负责返回chunk ID。
#
# Evidence全文由Python直接从原论文取出。
# =====================================================

def validate_and_enrich_item(
    item,
    chunk_map,
    source_file,
    source_sha1
):

    if not isinstance(
        item,
        dict
    ):

        return None, "item_not_dict"


    instruction = str(

        item.get(
            "instruction",
            ""
        )

    ).strip()


    output = str(

        item.get(
            "output",
            ""
        )

    ).strip()


    qa_type = str(

        item.get(
            "qa_type",
            ""
        )

    ).strip()


    evidence_chunk_ids = item.get(

        "evidence_chunk_ids",

        []

    )


    # =================================================
    # Question
    # =================================================

    if len(
        instruction
    ) < 20:

        return None, "question_too_short"


    # =================================================
    # Answer
    # =================================================

    if not output:

        return None, "empty_answer"


    answer_words = word_count(
        output
    )


    if (
        answer_words
        <
        MIN_ANSWER_WORDS
    ):

        return None, "answer_too_short"


    if (
        answer_words
        >
        MAX_ANSWER_WORDS
    ):

        return None, "answer_too_long"


    # =================================================
    # QA Type
    # =================================================

    if qa_type not in VALID_QA_TYPES:

        return None, "invalid_qa_type"


    # =================================================
    # 来源依赖语言
    # =================================================

    combined_text = (

        instruction
        +
        "\n"
        +
        output

    ).lower()


    for phrase in FORBIDDEN_PHRASES:

        if phrase in combined_text:

            return None, "source_dependent_language"


    # =================================================
    # Evidence IDs
    # =================================================

    if not isinstance(
        evidence_chunk_ids,
        list
    ):

        return None, "evidence_ids_not_list"


    if not evidence_chunk_ids:

        return None, "no_evidence"


    # 去重但保持顺序
    selected_chunk_ids = []

    for chunk_id in evidence_chunk_ids:

        chunk_id = str(
            chunk_id
        ).strip()

        if not chunk_id:

            continue

        if chunk_id not in selected_chunk_ids:

            selected_chunk_ids.append(
                chunk_id
            )


    if not selected_chunk_ids:

        return None, "no_evidence"


    if len(
        selected_chunk_ids
    ) > MAX_EVIDENCE_CHUNKS:

        return None, "too_many_evidence_chunks"


    # =================================================
    # Chunk ID真实性验证
    # =================================================

    for chunk_id in selected_chunk_ids:

        if chunk_id not in chunk_map:

            return None, "chunk_id_not_found"


    # =================================================
    # Python回填真正原始Evidence
    # =================================================

    evidence_chunks = []

    for chunk_id in selected_chunk_ids:

        source_chunk = chunk_map[
            chunk_id
        ]

        evidence_chunks.append(
            {
                "chunk_id":
                    chunk_id,

                "start_char":
                    int(
                        source_chunk[
                            "start_char"
                        ]
                    ),

                "end_char":
                    int(
                        source_chunk[
                            "end_char"
                        ]
                    ),

                "chunk_sha1":
                    source_chunk[
                        "chunk_sha1"
                    ],

                "text":
                    source_chunk[
                        "text"
                    ]
            }
        )


    # =================================================
    # Evidence长度
    # =================================================

    total_evidence_chars = sum(

        len(
            chunk[
                "text"
            ]
        )

        for chunk in evidence_chunks

    )


    if (
        total_evidence_chars
        >
        MAX_EVIDENCE_CHARS
    ):

        return None, "evidence_too_long"


    # =================================================
    # Evidence Text
    # =================================================

    evidence_text_parts = []

    for chunk in evidence_chunks:

        evidence_text_parts.append(

            f"[{chunk['chunk_id']}]\n"
            f"{chunk['text']}"

        )


    evidence_text = "\n\n".join(
        evidence_text_parts
    )


    # =================================================
    # 数字事实硬验证
    #
    # Answer出现的数字必须能在Evidence找到。
    # =================================================

    if not numeric_claims_supported(

        output,

        evidence_text

    ):

        return None, "unsupported_numeric_claim"


    # =================================================
    # 最终样本
    # =================================================

    enriched = {

        "instruction":
            instruction,

        "input":
            "",

        "output":
            output,

        "qa_type":
            qa_type,


        # ---------------------------------------------
        # Source traceability
        # ---------------------------------------------

        "source_file":
            source_file,

        "source_sha1":
            source_sha1,


        # ---------------------------------------------
        # Evidence traceability
        # ---------------------------------------------

        "evidence_chunk_ids":
            selected_chunk_ids,

        "evidence_chunks":
            evidence_chunks,

        "evidence_text":
            evidence_text,


        # ---------------------------------------------
        # Chunk配置
        # ---------------------------------------------

        "chunk_size":
            CHUNK_SIZE,

        "chunk_overlap":
            OVERLAP,


        # ---------------------------------------------
        # 统计
        # ---------------------------------------------

        "evidence_char_count":
            total_evidence_chars,

        "answer_word_count":
            answer_words,


        # ---------------------------------------------
        # 蒸馏信息
        # ---------------------------------------------

        "generator_model":
            MODEL,

        "thinking_enabled":
            ENABLE_THINKING,

        "schema_version":
            "rag_sft_v4_grounded_chunk_ids"

    }


    return enriched, None


# =====================================================
# 保存无法解析的原始返回
# =====================================================

def save_raw_response(
    source_file,
    retry,
    result_text
):

    name_hash = hashlib.sha1(

        source_file.encode(
            "utf-8",
            errors="ignore"
        )

    ).hexdigest()[
        :12
    ]


    save_file = (

        RAW_RESPONSE_DIR
        /
        f"{name_hash}_retry{retry}.txt"

    )


    try:

        save_file.write_text(

            str(result_text),

            encoding="utf-8"

        )

    except Exception:

        pass


# =====================================================
# API调用
# =====================================================

def call_flash(
    user_prompt
):

    extra_body = {}


    if ENABLE_THINKING:

        extra_body[
            "thinking"
        ] = {
            "type":
                "enabled"
        }


    response = (
        client
        .chat
        .completions
        .create(

            model=MODEL,

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT
                },

                {
                    "role":
                        "user",

                    "content":
                        user_prompt
                }

            ],


            # =========================================
            # 强制合法JSON输出
            # =========================================

            response_format={
                "type":
                    "json_object"
            },


            # =========================================
            # 开启DeepSeek thinking
            # =========================================

            extra_body=
                extra_body,


            # thinking模式下不再传temperature
            # 避免产生“温度参数有效”的误解


            max_tokens=
                MAX_OUTPUT_TOKENS,


            stream=False

        )
    )


    return response


# =====================================================
# 单篇论文生成
# =====================================================

def generate_one(
    path
):

    path = Path(
        path
    )


    # =================================================
    # Temporal protection
    # =================================================

    if contains_future_year(
        path.name
    ):

        return (
            path.name,
            [],
            True,
            "future_year"
        )


    # =================================================
    # 读取论文
    # =================================================

    article_text = read_article(
        path
    )


    if len(
        article_text
    ) < 500:

        return (
            path.name,
            [],
            True,
            "too_short"
        )


    source_sha1 = sha1_text(
        article_text
    )


    # =================================================
    # Chunk
    # =================================================

    chunks = split_into_chunks(
        article_text
    )


    if not chunks:

        return (
            path.name,
            [],
            True,
            "no_chunks"
        )


    chunk_map = {

        chunk[
            "chunk_id"
        ]:
            chunk

        for chunk in chunks

    }


    chunked_article = format_chunks_for_api(
        chunks
    )


    # =================================================
    # User prompt
    # =================================================

    user_prompt = (

        "Create high-quality evidence-grounded CNT RAG-SFT "
        "training data from the following chunked scientific "
        "article.\n\n"

        "Select the supporting chunk IDs FIRST, then construct "
        "each standalone question and answer.\n\n"

        "Important requirements:\n"
        "1. Return only valid JSON.\n"
        "2. Return only exact chunk IDs shown below.\n"
        "3. Use no more than two evidence chunks for each QA.\n"
        "4. Every claim in the answer must be supported by the "
        "selected chunks.\n"
        "5. Do not copy chunk text into the JSON.\n"
        "6. Prefer 2-4 scientifically valuable and diverse QA "
        "pairs.\n\n"

        "SOURCE ARTICLE:\n"

        +
        chunked_article

    )


    # =================================================
    # Retry
    # =================================================

    last_error = None


    for retry in range(
        1,
        MAX_RETRIES + 1
    ):

        result_text = None

        try:

            response = call_flash(
                user_prompt
            )


            result_text = (

                response
                .choices[0]
                .message
                .content

            )


            # =========================================
            # JSON
            # =========================================

            try:

                data = parse_json_response(
                    result_text
                )

            except Exception:

                save_raw_response(

                    path.name,

                    retry,

                    result_text

                )

                raise


            raw_items = data.get(
                "data",
                []
            )


            if not isinstance(
                raw_items,
                list
            ):

                raise ValueError(
                    "JSON中的data不是list"
                )


            # =========================================
            # 模型认为没有高价值内容
            # =========================================

            if len(
                raw_items
            ) == 0:

                return (
                    path.name,
                    [],
                    True,
                    "no_useful_qa"
                )


            valid_results = []

            rejection_reasons = []

            article_questions = set()


            # =========================================
            # 本地硬验证
            # =========================================

            for item in raw_items:

                enriched, reason = (
                    validate_and_enrich_item(

                        item=
                            item,

                        chunk_map=
                            chunk_map,

                        source_file=
                            path.name,

                        source_sha1=
                            source_sha1

                    )
                )


                if enriched is None:

                    rejection_reasons.append(
                        reason
                    )

                    continue


                q_key = normalize_question(

                    enriched[
                        "instruction"
                    ]

                )


                if q_key in article_questions:

                    rejection_reasons.append(
                        "duplicate_question"
                    )

                    continue


                article_questions.add(
                    q_key
                )


                valid_results.append(
                    enriched
                )


            # =========================================
            # 最多4条
            # =========================================

            valid_results = valid_results[
                :MAX_QA_PER_PAPER
            ]


            # =========================================
            # 如果全部被淘汰，则重试
            # =========================================

            if (
                raw_items
                and
                not valid_results
            ):

                reason_text = ",".join(

                    str(x)

                    for x in
                    rejection_reasons[
                        :10
                    ]

                )


                raise ValueError(

                    "所有候选QA均未通过本地验证: "
                    +
                    reason_text

                )


            # =========================================
            # 有效
            # =========================================

            return (
                path.name,
                valid_results,
                True,
                "ok"
            )


        except Exception as e:

            last_error = str(
                e
            )


            print(

                f"\n失败 [{path.name}] "
                f"retry={retry}: "
                f"{last_error[:400]}"

            )


            if retry < MAX_RETRIES:

                time.sleep(
                    3 * retry
                )


    # =================================================
    # 所有尝试失败
    # =================================================

    try:

        fail_file = (

            FAILED_DIR
            /
            (
                hashlib.sha1(
                    path.name.encode(
                        "utf-8",
                        errors="ignore"
                    )
                ).hexdigest()[
                    :12
                ]
                +
                ".txt"
            )

        )


        fail_file.write_text(

            (
                f"SOURCE_FILE:\n{path.name}\n\n"
                f"LAST_ERROR:\n{last_error}\n\n"
                f"ARTICLE:\n{article_text}"
            ),

            encoding="utf-8"

        )

    except Exception:

        pass


    return (
        path.name,
        [],
        False,
        "api_or_validation_failed"
    )


# =====================================================
# 已处理论文
# =====================================================

def load_processed_files():

    if not PROCESSED_FILE.exists():

        return set()


    return set(

        line.strip()

        for line in
        PROCESSED_FILE.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

        if line.strip()

    )


# =====================================================
# 已有Question
# =====================================================

def load_existing_questions():

    questions = set()


    if not OUTPUT_FILE.exists():

        return questions


    with open(

        OUTPUT_FILE,

        "r",

        encoding="utf-8"

    ) as f:


        for line in f:

            line = line.strip()

            if not line:

                continue


            try:

                item = json.loads(
                    line
                )

            except Exception:

                continue


            question = item.get(
                "instruction",
                ""
            )


            if question:

                questions.add(

                    normalize_question(
                        question
                    )

                )


    return questions


# =====================================================
# Main
# =====================================================

def main():


    print(
        "================================="
    )

    print(
        "YanlongGPT-CNT v4 Grounded RAG-SFT Generator"
    )

    print(
        "================================="
    )


    print(
        "模型:",
        MODEL
    )


    print(
        "API:",
        BASE_URL
    )


    print(
        "Thinking:",
        ENABLE_THINKING
    )


    print(
        "JSON Output:",
        True
    )


    print(
        "输入:",
        INPUT_DIR
    )


    print(
        "输出:",
        OUTPUT_FILE
    )


    print(
        "训练截止年份:",
        TRAIN_CUTOFF_YEAR
    )


    print(
        "Chunk size:",
        CHUNK_SIZE
    )


    print(
        "Chunk overlap:",
        OVERLAP
    )


    print(
        "Max evidence chunks:",
        MAX_EVIDENCE_CHUNKS
    )


    print(
        "Answer words:",
        f"{MIN_ANSWER_WORDS}-{MAX_ANSWER_WORDS}"
    )


    print(
        "并发数:",
        MAX_WORKERS
    )


    print(
        "单篇最大尝试:",
        MAX_RETRIES
    )


    # =================================================
    # 文件
    # =================================================

    files = sorted(

        INPUT_DIR.glob(
            "*.md"
        )

    )


    print(
        "\n原始论文数量:",
        len(files)
    )


    # =================================================
    # Temporal isolation
    # =================================================

    future_files = [

        file

        for file in files

        if contains_future_year(
            file.name
        )

    ]


    historical_files = [

        file

        for file in files

        if not contains_future_year(
            file.name
        )

    ]


    print(
        "Temporal过滤论文:",
        len(
            future_files
        )
    )


    print(
        "<=2024候选论文:",
        len(
            historical_files
        )
    )


    if future_files:

        with open(

            SKIPPED_FUTURE_FILE,

            "w",

            encoding="utf-8"

        ) as f:


            for file in future_files:

                f.write(
                    file.name
                    +
                    "\n"
                )


    files = historical_files


    # =================================================
    # Resume
    # =================================================

    processed_files = (
        load_processed_files()
    )


    existing_questions = (
        load_existing_questions()
    )


    files = [

        file

        for file in files

        if file.name
        not in processed_files

    ]


    print(
        "已经处理:",
        len(
            processed_files
        )
    )


    print(
        "剩余:",
        len(
            files
        )
    )


    # =================================================
    # Test
    # =================================================

    if TEST_LIMIT is not None:

        files = files[
            :TEST_LIMIT
        ]


        print(
            "\n当前TEST模式，只处理:",
            len(
                files
            ),
            "篇"
        )


    if not files:

        print(
            "没有需要处理的论文"
        )

        return


    # =================================================
    # 并发
    # =================================================

    total_written = 0

    total_success_papers = 0

    total_zero_qa_papers = 0

    total_failed_papers = 0


    with ThreadPoolExecutor(

        max_workers=
            MAX_WORKERS

    ) as executor:


        futures = [

            executor.submit(

                generate_one,

                str(
                    file
                )

            )

            for file in files

        ]


        with open(

            OUTPUT_FILE,

            "a",

            encoding="utf-8"

        ) as out, open(

            PROCESSED_FILE,

            "a",

            encoding="utf-8"

        ) as processed_out:


            for future in tqdm(

                as_completed(
                    futures
                ),

                total=len(
                    futures
                )

            ):


                try:

                    (
                        source_file,
                        results,
                        success,
                        status

                    ) = future.result()


                except Exception as e:

                    print(
                        "\nFuture失败:",
                        str(e)[:400]
                    )

                    total_failed_papers += 1

                    continue


                # =========================================
                # 完全失败不写processed
                # =========================================

                if not success:

                    total_failed_papers += 1

                    continue


                paper_written = 0


                # =========================================
                # 写入QA
                # =========================================

                for item in results:


                    question_key = (
                        normalize_question(

                            item[
                                "instruction"
                            ]

                        )
                    )


                    if (
                        question_key
                        in existing_questions
                    ):

                        continue


                    out.write(

                        json.dumps(

                            item,

                            ensure_ascii=False

                        )

                        +
                        "\n"

                    )


                    out.flush()


                    existing_questions.add(
                        question_key
                    )


                    paper_written += 1

                    total_written += 1


                # =========================================
                # API正常完成才写processed
                # =========================================

                processed_out.write(

                    source_file
                    +
                    "\n"

                )


                processed_out.flush()


                total_success_papers += 1


                if paper_written == 0:

                    total_zero_qa_papers += 1


                print(

                    f"\n{source_file}"
                    f" -> 保存 {paper_written} 条Grounded QA"
                    f" | status={status}"

                )


    # =================================================
    # 最终统计
    # =================================================

    print(
        "\n================================="
    )

    print(
        "Grounded RAG-SFT生成完成"
    )

    print(
        "================================="
    )


    print(
        "成功处理论文:",
        total_success_papers
    )


    print(
        "0条QA论文:",
        total_zero_qa_papers
    )


    print(
        "彻底失败论文:",
        total_failed_papers
    )


    print(
        "本次新增Grounded QA:",
        total_written
    )


    print(
        "\n输出:"
    )

    print(
        OUTPUT_FILE
    )


    print(
        "\nProcessed:"
    )

    print(
        PROCESSED_FILE
    )


    print(
        "\n失败API原始返回:"
    )

    print(
        RAW_RESPONSE_DIR
    )


    print(
        "\n下一步："
    )

    print(
        "先检查20篇生成的数据质量"
    )


    print(
        "================================="
    )


if __name__ == "__main__":

    main()