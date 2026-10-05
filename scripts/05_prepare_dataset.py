from pathlib import Path
import random
import numpy as np
from tokenizers import Tokenizer


# =========================
# 路径设置
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent


CORPUS_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "cnt_corpus.txt"
)


TOKENIZER_FILE = (
    BASE_DIR
    / "tokenizer"
    / "cnt_tokenizer.json"
)


OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "train"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# 参数
# =========================

# 验证集论文比例
VAL_RATIO = 0.05

# 固定随机种子，保证每次划分一致
RANDOM_SEED = 42


# =========================
# 主程序
# =========================

def main():

    # -------------------------
    # 加载 tokenizer
    # -------------------------

    print("加载 tokenizer...")

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_FILE)
    )

    vocab_size = tokenizer.get_vocab_size()

    print(
        "词表大小:",
        vocab_size
    )


    # uint16 最大可以表示 65535
    # 当前 vocab=10000，完全安全
    if vocab_size >= 65536:
        raise ValueError(
            "词表大小超过 uint16 范围，需要改用 uint32"
        )


    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")


    if bos_id is None or eos_id is None:
        raise ValueError(
            "Tokenizer 中缺少 [BOS] 或 [EOS]"
        )


    print(
        "BOS ID:",
        bos_id
    )

    print(
        "EOS ID:",
        eos_id
    )


    # -------------------------
    # 读取语料
    # -------------------------

    print("\n读取语料...")

    text = CORPUS_FILE.read_text(
        encoding="utf-8",
        errors="ignore"
    )


    print(
        "字符数:",
        len(text)
    )


    # -------------------------
    # 按文章拆分
    # -------------------------

    raw_articles = text.split("<CNT_ARTICLE>")


    articles = []


    for article in raw_articles:

        article = article.replace(
            "</CNT_ARTICLE>",
            ""
        ).strip()


        if article:

            articles.append(article)


    print(
        "识别到论文数量:",
        len(articles)
    )


    if len(articles) == 0:
        raise ValueError(
            "没有成功识别任何论文，请检查 corpus 格式"
        )


    # -------------------------
    # 论文级随机划分
    # -------------------------

    rng = random.Random(
        RANDOM_SEED
    )


    rng.shuffle(
        articles
    )


    val_count = max(
        1,
        int(len(articles) * VAL_RATIO)
    )


    val_articles = articles[:val_count]

    train_articles = articles[val_count:]


    print("\n====================")

    print(
        "训练论文:",
        len(train_articles)
    )

    print(
        "验证论文:",
        len(val_articles)
    )

    print("====================")


    # -------------------------
    # Tokenize函数
    # -------------------------

    def encode_articles(article_list, name):

        all_ids = []


        print(
            f"\n开始 tokenize {name}..."
        )


        total = len(article_list)


        for i, article in enumerate(article_list):

            ids = tokenizer.encode(
                article
            ).ids


            # 每篇论文：
            # [BOS] + 正文 + [EOS]

            all_ids.append(
                bos_id
            )

            all_ids.extend(
                ids
            )

            all_ids.append(
                eos_id
            )


            if (
                (i + 1) % 100 == 0
                or
                (i + 1) == total
            ):

                print(
                    f"[{i + 1}/{total}]"
                )


        return np.array(
            all_ids,
            dtype=np.uint16
        )


    # -------------------------
    # Tokenize
    # -------------------------

    train_ids = encode_articles(
        train_articles,
        "训练集"
    )


    val_ids = encode_articles(
        val_articles,
        "验证集"
    )


    # -------------------------
    # 统计
    # -------------------------

    total_tokens = (
        len(train_ids)
        +
        len(val_ids)
    )


    print("\n====================")

    print(
        "总 token:",
        total_tokens
    )

    print(
        "训练 token:",
        len(train_ids)
    )

    print(
        "验证 token:",
        len(val_ids)
    )


    print(
        "训练比例:",
        round(
            len(train_ids)
            /
            total_tokens,
            4
        )
    )


    print(
        "验证比例:",
        round(
            len(val_ids)
            /
            total_tokens,
            4
        )
    )

    print("====================")


    # -------------------------
    # 保存
    # -------------------------

    train_file = (
        OUTPUT_DIR
        /
        "train.bin"
    )

    val_file = (
        OUTPUT_DIR
        /
        "val.bin"
    )


    train_ids.tofile(
        train_file
    )


    val_ids.tofile(
        val_file
    )


    print("\n====================")

    print(
        "训练集保存:",
        train_file
    )

    print(
        "验证集保存:",
        val_file
    )

    print("====================")


if __name__ == "__main__":

    main()