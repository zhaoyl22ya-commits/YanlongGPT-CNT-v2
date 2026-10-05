import json
import re
from pathlib import Path
from collections import Counter


# =====================================================
# 路径
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    /
    "data"
    /
    "sft"
    /
    "cnt_sft_dataset.jsonl"
)


OUTPUT_FILE = (
    BASE_DIR
    /
    "data"
    /
    "sft"
    /
    "cnt_sft_dataset_clean.jsonl"
)


# =====================================================
# 参数
# =====================================================

# 这里是“宽松清洗”，不是严格卡死生成Prompt的60-160 words
MIN_OUTPUT_LEN = 180
MAX_OUTPUT_LEN = 1800

MIN_INSTRUCTION_LEN = 15
MAX_INSTRUCTION_LEN = 500


# =====================================================
# 合法QA类型
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
# 禁止的论文复述式表达
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
# 低价值百科问题
# =====================================================

BAD_QUESTIONS = [

    "what is cnt",

    "what are cnts",

    "what is a carbon nanotube",

    "what are carbon nanotubes",

    "what is cvd",

    "what are cnts used for",

    "what are carbon nanotubes used for",

    "introduce carbon nanotubes"
]


# =====================================================
# 文本规范
# =====================================================

def clean_text(text):

    if not isinstance(text, str):

        return ""


    # 去首尾空格
    text = text.strip()


    # 多空格 / 换行统一
    text = re.sub(

        r"\s+",

        " ",

        text
    )


    return text


# =====================================================
# 判断是否有科研价值
# =====================================================

def is_high_quality(item):

    q = item["instruction"]

    a = item["output"]


    # ===============================
    # 问题长度
    # ===============================

    if len(q) < MIN_INSTRUCTION_LEN:

        return False


    if len(q) > MAX_INSTRUCTION_LEN:

        return False


    # ===============================
    # 回答长度
    # ===============================

    if len(a) < MIN_OUTPUT_LEN:

        return False


    if len(a) > MAX_OUTPUT_LEN:

        return False


    # ===============================
    # 删除百科问题
    # ===============================

    q_lower = q.lower()


    for bad in BAD_QUESTIONS:

        if bad in q_lower:

            return False


    # ===============================
    # 删除来源依赖型表达
    # ===============================

    combined = (

        q

        +

        " "

        +

        a

    ).lower()


    for phrase in FORBIDDEN_PHRASES:

        if phrase in combined:

            return False


    # ===============================
    # QA类型检查
    # ===============================

    qa_type = item.get(
        "qa_type",
        ""
    )


    if qa_type not in VALID_QA_TYPES:

        return False


    return True


# =====================================================
# 问题标准化，用于去重
# =====================================================

def normalize_question(text):

    text = text.lower().strip()


    # 去掉标点差异
    text = re.sub(

        r"[^a-z0-9]+",

        " ",

        text
    )


    # 多空格统一
    text = re.sub(

        r"\s+",

        " ",

        text
    )


    return text.strip()


# =====================================================
# 主程序
# =====================================================

def main():


    print("==============================")

    print("YanlongGPT-CNT v2 SFT Cleaner")

    print("==============================")


    print("读取:")

    print(INPUT_FILE)


    if not INPUT_FILE.exists():

        raise FileNotFoundError(

            f"找不到SFT文件: {INPUT_FILE}"
        )


    # =================================================
    # 读取
    # =================================================

    data = []

    json_failed = 0


    with open(

        INPUT_FILE,

        "r",

        encoding="utf-8"

    ) as f:


        for line in f:


            line = line.strip()


            if not line:

                continue


            try:

                item = json.loads(line)

                data.append(item)


            except Exception:

                json_failed += 1


    print(

        "原始数量:",

        len(data)

    )


    print(

        "JSON解析失败:",

        json_failed

    )


    # =================================================
    # 清洗
    # =================================================

    cleaned = []


    reject_counter = Counter()


    for item in data:


        # =============================
        # 必须字段
        # =============================

        if not all(

            k in item

            for k in [

                "instruction",

                "output",

                "qa_type",

                "source_file"

            ]

        ):

            reject_counter["missing_field"] += 1

            continue


        # =============================
        # 文本清洗
        # =============================

        item["instruction"] = clean_text(

            item["instruction"]

        )


        item["output"] = clean_text(

            item["output"]

        )


        # input统一为空
        item["input"] = ""


        # =============================
        # 基本质量过滤
        # =============================

        if not is_high_quality(item):

            reject_counter["quality_filter"] += 1

            continue


        # 注意：
        # v2不再强制要求问题/答案里必须出现CNT关键词。
        #
        # 因为XPS、场发射、催化剂、界面、
        # 复合材料等问题虽然不一定直接出现CNT，
        # 但仍然属于CNT论文中的有效领域知识。


        cleaned.append(item)


    print(

        "质量过滤后:",

        len(cleaned)

    )


    # =================================================
    # 去重
    # =================================================

    unique = []

    seen = set()

    duplicate_count = 0


    for item in cleaned:


        key = normalize_question(

            item["instruction"]

        )


        if key in seen:

            duplicate_count += 1

            continue


        seen.add(key)

        unique.append(item)


    print(

        "重复问题删除:",

        duplicate_count

    )


    print(

        "最终数量:",

        len(unique)

    )


    # =================================================
    # 保存
    # =================================================

    OUTPUT_FILE.parent.mkdir(

        parents=True,

        exist_ok=True

    )


    with open(

        OUTPUT_FILE,

        "w",

        encoding="utf-8"

    ) as f:


        for item in unique:


            f.write(

                json.dumps(

                    item,

                    ensure_ascii=False

                )

                +

                "\n"

            )


    # =================================================
    # QA类型统计
    # =================================================

    qa_counter = Counter(

        item["qa_type"]

        for item in unique

    )


    print("\n================")

    print("QA类型分布")

    print("================")


    for qa_type, count in (

        qa_counter.most_common()

    ):

        print(

            qa_type,

            count

        )


    # =================================================
    # 论文覆盖统计
    # =================================================

    source_counter = Counter(

        item["source_file"]

        for item in unique

    )


    print("\n================")

    print("论文覆盖")

    print("================")


    print(

        "包含QA的论文数量:",

        len(source_counter)

    )


    if source_counter:


        counts = list(

            source_counter.values()

        )


        print(

            "平均每篇QA:",

            round(

                sum(counts)

                /

                len(counts),

                2

            )

        )


        print(

            "最少每篇QA:",

            min(counts)

        )


        print(

            "最多每篇QA:",

            max(counts)

        )


    print("\n================")

    print("清洗完成")

    print(OUTPUT_FILE)

    print("================")


if __name__ == "__main__":

    main()