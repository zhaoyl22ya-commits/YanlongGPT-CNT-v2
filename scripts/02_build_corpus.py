from pathlib import Path


# =========================
# 路径设置
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent


INPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "CNT_clean"
)


OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cnt_corpus.txt"
)



# =========================
# 合并函数
# =========================

def main():

    files = sorted(
        INPUT_DIR.glob("*.md")
    )


    print(
        f"发现 {len(files)} 篇清洗后的论文"
    )


    count = 0


    total_chars = 0


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:


        for file in files:


            try:

                text = file.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )


                # 添加文章分隔符
                outfile.write(
                    "\n\n<CNT_ARTICLE>\n\n"
                )


                # 保留原文本
                outfile.write(text)


                outfile.write(
                    "\n\n</CNT_ARTICLE>\n\n"
                )


                count += 1

                total_chars += len(text)



            except Exception as e:

                print(
                    "读取失败:",
                    file.name,
                    e
                )



    print("========================")

    print(
        "成功合并文章:",
        count
    )


    print(
        "总字符数:",
        total_chars
    )


    print(
        "输出文件:"
    )


    print(
        OUTPUT_FILE
    )



if __name__ == "__main__":

    main()