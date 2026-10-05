from pathlib import Path
import os


# =========================
# 路径设置
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent


CORPUS_FILE = (
    BASE_DIR
    /
    "data"
    /
    "processed"
    /
    "cnt_corpus.txt"
)



def main():

    print("正在读取语料...")


    text = CORPUS_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )


    chars = len(text)


    lines = text.count("\n")


    words = len(
        text.split()
    )


    size_mb = (
        os.path.getsize(CORPUS_FILE)
        /
        1024
        /
        1024
    )


    print("====================")

    print(
        "文件大小:",
        round(size_mb,2),
        "MB"
    )


    print(
        "字符数量:",
        chars
    )


    print(
        "单词数量:",
        words
    )


    print(
        "行数:",
        lines
    )


    print("====================")


    # 粗略英文token估算
    estimated_tokens = int(
        words * 1.3
    )


    print(
        "估算token数量:",
        estimated_tokens
    )



if __name__=="__main__":

    main()