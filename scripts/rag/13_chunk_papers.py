from pathlib import Path
import json
import re


# =========================
# 路径
# =========================

# 当前文件：
# 项目根目录/scripts/13_chunk_papers.py
# parent.parent = 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]


# 2025–2026 文献专用 RAG 语料目录
PAPER_DIR = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "papers_2025_2026"
)


# chunk 输出
SAVE_FILE = (
    BASE_DIR
    /
    "data"
    /
    "rag"
    /
    "cnt_chunks.jsonl"
)


SAVE_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# 参数
# =========================

# 按字符切分
CHUNK_SIZE = 1000

# 相邻 chunk 重叠字符数
OVERLAP = 200


# RAG只允许2025–2026论文
ALLOWED_YEARS = {
    "2025",
    "2026"
}


# =========================
# 年份检查
# =========================

def get_allowed_year(
    filename
):

    """
    从文件名中识别年份。

    只接受：
    2025
    2026

    其他年份或无法识别年份的文件返回 None。
    """

    years = re.findall(
        r"(?<!\d)(20\d{2})(?!\d)",
        filename
    )


    for year in years:

        if year in ALLOWED_YEARS:

            return year


    return None


# =========================
# chunk函数
# =========================

def split_text(
    text
):


    chunks = []


    start = 0


    length = len(text)


    while start < length:


        end = start + CHUNK_SIZE


        chunk = text[
            start:end
        ]


        # 避免写入完全空白的chunk
        if chunk.strip():

            chunks.append(
                chunk
            )


        start = end - OVERLAP


    return chunks


# =========================
# main
# =========================

def main():


    # =========================
    # 检查输入目录
    # =========================

    if not PAPER_DIR.exists():

        raise FileNotFoundError(
            f"找不到RAG论文目录: {PAPER_DIR}"
        )


    # =========================
    # 获取所有Markdown
    # =========================

    all_md_files = sorted(
        PAPER_DIR.glob("*.md")
    )


    print(
        "RAG论文目录:"
    )

    print(
        PAPER_DIR
    )


    print(
        "\n目录内MD数量:",
        len(all_md_files)
    )


    # =========================
    # 年份过滤
    # =========================

    md_files = []

    skipped_files = []


    for file in all_md_files:


        year = get_allowed_year(
            file.name
        )


        if year is None:

            skipped_files.append(
                file
            )

        else:

            md_files.append(
                file
            )


    print(
        "2025–2026有效论文:",
        len(md_files)
    )


    print(
        "跳过非2025–2026论文:",
        len(skipped_files)
    )


    # 打印被跳过的文件
    if skipped_files:


        print(
            "\n以下文件未进入RAG:"
        )


        for file in skipped_files:

            print(
                "跳过:",
                file.name
            )


    if len(md_files) == 0:

        raise ValueError(
            "没有找到有效的2025–2026 Markdown论文"
        )


    # =========================
    # 开始chunk
    # =========================

    total = 0


    with open(
        SAVE_FILE,
        "w",
        encoding="utf-8"
    ) as fout:


        for paper_index, file in enumerate(
            md_files,
            start=1
        ):


            print(
                f"[{paper_index}/{len(md_files)}]",
                "处理:",
                file.name
            )


            text = file.read_text(
                encoding="utf-8",
                errors="ignore"
            )


            # 空文件跳过
            if not text.strip():

                print(
                    "空文件，跳过:",
                    file.name
                )

                continue


            chunks = split_text(
                text
            )


            for i, chunk in enumerate(
                chunks
            ):


                item = {

                    "id":
                        total,

                    "source":
                        file.name,

                    "chunk_id":
                        i,

                    "text":
                        chunk

                }


                fout.write(

                    json.dumps(
                        item,
                        ensure_ascii=False
                    )

                    +

                    "\n"

                )


                total += 1


    # =========================
    # 完成
    # =========================

    print(
        "\n================"
    )


    print(
        "chunk完成"
    )


    print(
        "有效论文:",
        len(md_files)
    )


    print(
        "跳过论文:",
        len(skipped_files)
    )


    print(
        "总chunk:",
        total
    )


    print(
        "保存位置:"
    )


    print(
        SAVE_FILE
    )


    print(
        "================"
    )


if __name__ == "__main__":

    main()