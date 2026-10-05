import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm


# =====================================================
# 项目路径
# =====================================================

# 当前文件：
# 项目根目录/scripts/08_generate_cnt_sft.py
BASE_DIR = Path(__file__).resolve().parent.parent

SCRIPT_DIR = Path(__file__).resolve().parent


# =====================================================
# API 配置
# =====================================================

KEY_FILE = SCRIPT_DIR / "key"

MODEL = "DeepSeek-V4-Pro-0813"

BASE_URL = "https://llmapi.paratera.com/v1/"


def load_api_key():

    if not KEY_FILE.exists():

        raise FileNotFoundError(
            f"找不到 API 配置文件: {KEY_FILE}"
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


        # 支持：
        # API_KEY: sk-xxx
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


        # 也支持 key 文件中直接只有一行 sk-xxx
        elif line.startswith("sk-"):

            api_key = line


    if api_key is None:

        raise ValueError(
            "key 文件中没有找到 API_KEY"
        )


    return api_key


API_KEY = load_api_key()


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=180.0
)


# =====================================================
# 数据路径
# =====================================================

INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "CNT_clean"
)


OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "sft"
    / "cnt_sft_dataset.jsonl"
)


FAILED_DIR = (
    BASE_DIR
    / "data"
    / "sft"
    / "failed"
)


PROCESSED_FILE = (
    BASE_DIR
    / "data"
    / "sft"
    / "processed_files.txt"
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


FAILED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# 参数
# =====================================================

# API并行数
# 当前平台RPM/TPM很高，12个并发比较保守
MAX_WORKERS = 48


# DeepSeek-V4-Pro-0813上下文很长，
# 不再像旧版一样只截前40000字符。
# 300000基本可以覆盖绝大部分完整论文。
MAX_CHARS = 300000


# 单次最大输出
MAX_OUTPUT_TOKENS = 3000


# 生成温度：
# 保证稳定性的同时保留一定问法多样性
TEMPERATURE = 0.25


# =====================================================
# 测试模式
# =====================================================

# 第一次建议设为5：
# 只跑5篇检查QA质量。
#
# 确认效果满意以后改成：
#
# TEST_LIMIT = None
#
# 就会处理全部论文。

TEST_LIMIT = None


# =====================================================
# QA 类型
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
# 禁止出现的模板表达
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

    "the paper reports"

]


# =====================================================
# SFT Prompt
# =====================================================

SYSTEM_PROMPT = r"""
You are an expert in carbon nanotube (CNT) synthesis,
catalytic chemical vapor deposition, catalyst chemistry,
nanomaterial characterization, and process optimization.

Your task is to create HIGH-QUALITY supervised fine-tuning
data for a domain-specific CNT scientific assistant.

You will receive the text of ONE scientific article.

Generate 3 to 5 independent scientific
Instruction-Response pairs from the article.

The goal is NOT to summarize the article.

The goal is to transform scientifically useful knowledge
contained in the article into standalone questions and
direct expert answers that can train a CNT research assistant.


=========================================================
1. SCIENTIFIC ACCURACY
=========================================================

Every factual claim in the answer must be supported by
the supplied article.

Do NOT invent:

- experimental conditions
- temperatures
- pressures
- gas flow rates
- catalyst compositions
- particle sizes
- CNT diameters
- Raman values
- yields
- mechanisms
- numerical results

If the source article does not support a claim,
do not include it.

If the article contains insufficient useful CNT synthesis
information, return fewer than 5 question-answer pairs.

Quality is more important than quantity.


=========================================================
2. QUESTION INDEPENDENCE
=========================================================

Every question must be understandable WITHOUT access
to the original article.

BAD questions:

"What catalyst was used in this paper?"

"What did the authors find?"

"What was the optimal temperature in this study?"

"Why was this method selected?"

GOOD questions:

"How does Fe catalyst particle size influence CNT diameter
during chemical vapor deposition?"

"Why can excessive hydrocarbon concentration suppress
vertically aligned CNT forest growth?"

"What does an increased Raman ID/IG ratio indicate about
structural disorder in carbon nanotubes?"

When the answer depends on a specific experimental system,
include that system directly in the question.


=========================================================
3. ANSWER STYLE
=========================================================

Answers must immediately answer the scientific question.

Never begin with or rely on phrases such as:

"According to the paper"
"This paper"
"This study"
"In this study"
"The authors"
"The researchers"
"The article"
"In this work"
"The present work"

Do NOT describe the source article as an article.

Do NOT say that someone "reported", "found", or "demonstrated"
something unless attribution itself is scientifically necessary.

Write the answer as domain knowledge.

For example:

BAD:

"This paper reports that smaller Fe nanoparticles resulted
in smaller CNT diameters."

GOOD:

"Smaller Fe nanoparticles can favor the formation of
smaller-diameter CNTs because the catalyst particle defines
the characteristic nucleation scale during growth."


=========================================================
4. DIVERSITY WITHIN EACH ARTICLE
=========================================================

The 3-5 questions generated from one article MUST cover
different scientific aspects whenever the article allows it.

Do NOT generate several questions that test essentially
the same fact using different wording.

Prefer distinct high-value question types such as:

- growth mechanism
- catalyst composition or catalyst particle-size effect
- catalyst support or promoter effect
- temperature or pressure effect
- carbon source or gas-flow effect
- synthesis strategy
- CNT morphology or structural control
- Raman / TEM / SEM / XRD interpretation
- structure-property relationship
- quantitative experimental condition
- comparison between synthesis conditions
- process optimization
- growth failure or limitation mechanism

For 4-5 generated pairs, use at least 4 distinct qa_type
values whenever scientifically possible.

Do not ask two questions whose answers are essentially
the same scientific statement.


=========================================================
5. QUESTION VALUE
=========================================================

Prefer questions that a CNT researcher may genuinely ask
during:

- experimental design
- catalyst selection
- process optimization
- mechanism analysis
- characterization interpretation
- troubleshooting
- synthesis comparison

Avoid low-value encyclopedia questions such as:

"What is a carbon nanotube?"

"What is CVD?"

"What are CNTs used for?"

Avoid questions that only test article metadata,
author names, journal names, section names, or publication year.


=========================================================
6. ANSWER LENGTH
=========================================================

Each answer should normally contain approximately
60-160 English words.

Use 2-5 concise scientific sentences.

Longer answers are allowed only when required to explain
a mechanism or a multi-variable relationship.

Preserve important quantitative experimental conditions
when they are scientifically meaningful.


=========================================================
7. LANGUAGE
=========================================================

Question and answer must both be written in English.

Use professional scientific language.

Avoid unnecessary verbosity.

Avoid generic introductory sentences.


=========================================================
8. QA TYPE
=========================================================

For every pair, provide one qa_type selected from:

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
9. OUTPUT FORMAT
=========================================================

Return ONLY valid JSON.

Do not use Markdown.

Use exactly this structure:

{
  "data": [
    {
      "instruction": "standalone scientific question",
      "input": "",
      "output": "direct scientific answer",
      "qa_type": "one allowed qa_type"
    }
  ]
}

Generate between 3 and 5 pairs when sufficient
scientific information exists.

Return fewer pairs if necessary rather than fabricating
or repeating information.
"""


# =====================================================
# 文本清洗
# =====================================================

def clean_text(text):

    replace_map = {

        "\x00": "",

        "\r": "",

        "–": "-",

        "—": "-",

        "−": "-",

        "×": "x",

        "℃": " °C",

        "α": "alpha",

        "β": "beta",

        "γ": "gamma",

        "μ": "micro",

        "≤": "<=",

        "≥": ">=",

        "±": "+/-"
    }


    for k, v in replace_map.items():

        text = text.replace(
            k,
            v
        )


    text = "".join(

        c for c in text

        if ord(c) >= 32

        or c in "\n\t"
    )


    return text


# =====================================================
# 读取论文
# =====================================================

def read_md(path):

    try:

        text = Path(path).read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception:

        return ""


    text = clean_text(
        text
    )


    # 极端异常超长文件才截断
    if len(text) > MAX_CHARS:

        text = text[:MAX_CHARS]


    return text


# =====================================================
# JSON解析
# =====================================================

def parse_json_response(text):

    text = text.strip()


    # 去掉可能出现的 Markdown JSON fence

    if text.startswith("```"):

        text = text.replace(
            "```json",
            "",
            1
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()


    # 尝试直接解析

    try:

        return json.loads(
            text
        )

    except Exception:

        pass


    # 如果模型额外输出少量文字，
    # 尝试提取第一个JSON对象

    start = text.find("{")

    end = text.rfind("}")


    if (
        start != -1
        and
        end != -1
        and
        end > start
    ):

        try:

            return json.loads(
                text[
                    start:
                    end + 1
                ]
            )

        except Exception:

            pass


    raise ValueError(
        "无法解析模型返回的JSON"
    )


# =====================================================
# QA质量过滤
# =====================================================

def valid_item(item):

    if not isinstance(
        item,
        dict
    ):

        return False


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


    if not instruction:

        return False


    if not output:

        return False


    if qa_type not in VALID_QA_TYPES:

        return False


    # 问题太短通常没有科研价值
    if len(instruction) < 20:

        return False


    # 回答过短
    if len(output) < 120:

        return False


    combined = (
        instruction
        +
        "\n"
        +
        output
    ).lower()


    # 删除来源依赖型问答
    for phrase in FORBIDDEN_PHRASES:

        if phrase in combined:

            return False


    return True


# =====================================================
# DeepSeek生成
# =====================================================

def generate_one(path):

    path = Path(path)


    text = read_md(
        path
    )


    # 正文太少
    if len(text) < 500:

        return (
            path.name,
            [],
            True
        )


    for retry in range(3):

        try:

            response = client.chat.completions.create(

                model=MODEL,


                messages=[

                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },

                    {
                        "role": "user",
                        "content": (
                            "Generate high-quality CNT SFT data "
                            "from the following scientific article:\n\n"
                            +
                            text
                        )
                    }

                ],


                temperature=TEMPERATURE,


                max_tokens=MAX_OUTPUT_TOKENS
            )


            result = (
                response
                .choices[0]
                .message
                .content
            )


            data = parse_json_response(
                result
            )


            items = data.get(
                "data",
                []
            )


            if not isinstance(
                items,
                list
            ):

                raise ValueError(
                    "data不是list"
                )


            valid_results = []


            for item in items:

                if not valid_item(
                    item
                ):

                    continue


                new_item = {

                    "instruction":
                        str(
                            item["instruction"]
                        ).strip(),

                    "input":
                        "",

                    "output":
                        str(
                            item["output"]
                        ).strip(),

                    "qa_type":
                        str(
                            item["qa_type"]
                        ).strip(),

                    # 仅用于追踪来源
                    # 后续SFT训练时不会喂给模型
                    "source_file":
                        path.name
                }


                valid_results.append(
                    new_item
                )


            # 最多5条
            valid_results = (
                valid_results[:5]
            )


            return (
                path.name,
                valid_results,
                True
            )


        except Exception as e:

            print(
                f"\n失败 [{path.name}] "
                f"retry={retry + 1}: "
                f"{str(e)[:200]}"
            )


            time.sleep(
                3 * (retry + 1)
            )


    # =================================================
    # 三次都失败
    # =================================================

    try:

        fail_file = (
            FAILED_DIR
            /
            (
                path.stem
                +
                "_failed.txt"
            )
        )


        fail_file.write_text(
            text,
            encoding="utf-8"
        )

    except Exception:

        pass


    return (
        path.name,
        [],
        False
    )


# =====================================================
# 已处理论文
# =====================================================

def load_processed_files():

    if not PROCESSED_FILE.exists():

        return set()


    return set(

        line.strip()

        for line in PROCESSED_FILE.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

        if line.strip()
    )


# =====================================================
# 已有Question去重
# =====================================================

def normalize_question(text):

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


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

            try:

                item = json.loads(
                    line
                )


                question = (
                    item.get(
                        "instruction",
                        ""
                    )
                )


                if question:

                    questions.add(
                        normalize_question(
                            question
                        )
                    )


            except Exception:

                continue


    return questions


# =====================================================
# 主程序
# =====================================================

def main():

    print(
        "================================="
    )

    print(
        "YanlongGPT-CNT v2 SFT Generator"
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
        "输入目录:",
        INPUT_DIR
    )


    print(
        "输出:",
        OUTPUT_FILE
    )


    print(
        "并发数:",
        MAX_WORKERS
    )


    files = sorted(
        INPUT_DIR.glob(
            "*.md"
        )
    )


    print(
        "论文总数:",
        len(files)
    )


    processed_files = (
        load_processed_files()
    )


    existing_questions = (
        load_existing_questions()
    )


    # 删除已成功处理论文
    files = [

        f for f in files

        if f.name
        not in processed_files

    ]


    print(
        "已经处理:",
        len(processed_files)
    )


    print(
        "剩余:",
        len(files)
    )


    # 测试模式
    if TEST_LIMIT is not None:

        files = files[
            :TEST_LIMIT
        ]

        print(
            "当前为测试模式，只处理:",
            len(files),
            "篇"
        )


    if not files:

        print(
            "没有需要处理的论文"
        )

        return


    total_written = 0


    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:


        futures = [

            executor.submit(
                generate_one,
                str(file)
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
                        success
                    ) = future.result()


                except Exception as e:

                    print(
                        "\nfuture失败:",
                        str(e)[:200]
                    )

                    continue


                # API调用失败：
                # 不标记processed，
                # 下次重新运行还会重试
                if not success:

                    continue


                paper_written = 0


                for item in results:


                    question_key = (
                        normalize_question(
                            item[
                                "instruction"
                            ]
                        )
                    )


                    # 全数据集Question去重
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


                # 即使论文因为内容不足只生成0条，
                # 只要API成功，也标记为已处理
                processed_out.write(
                    source_file
                    +
                    "\n"
                )

                processed_out.flush()


                print(
                    f"\n{source_file}"
                    f" -> 保存 {paper_written} 条QA"
                )


    print(
        "\n================================="
    )

    print(
        "SFT数据生成完成"
    )

    print(
        "本次新增QA:",
        total_written
    )

    print(
        "输出文件:",
        OUTPUT_FILE
    )

    print(
        "================================="
    )


if __name__ == "__main__":

    main()