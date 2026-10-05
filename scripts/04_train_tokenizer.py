from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# =========================
# 路径设置
# =========================

# 当前文件位于：
# 项目根目录/scripts/04_train_tokenizer.py
# 因此 parent.parent 为项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent


CORPUS = (
    BASE_DIR
    / "data"
    / "processed"
    / "cnt_corpus.txt"
)


SAVE_PATH = (
    BASE_DIR
    / "tokenizer"
    / "cnt_tokenizer.json"
)


SAVE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# Tokenizer 参数
# =========================

VOCAB_SIZE = 10000
MIN_FREQUENCY = 3


# =========================
# 主程序
# =========================

def main():

    print("========================")
    print("开始训练 CNT Tokenizer")
    print("语料文件:", CORPUS)
    print("词表大小:", VOCAB_SIZE)
    print("最低词频:", MIN_FREQUENCY)
    print("========================")


    if not CORPUS.exists():

        raise FileNotFoundError(
            f"找不到语料文件: {CORPUS}"
        )


    # =========================
    # BPE Tokenizer
    # =========================

    tokenizer = Tokenizer(
        BPE(
            unk_token="[UNK]"
        )
    )


    # ByteLevel：
    # 比原来的 Whitespace 更适合当前英文科学语料
    # 同时能够正确处理空格、标点、数字和特殊字符
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )


    tokenizer.decoder = ByteLevelDecoder()


    # =========================
    # BPE Trainer
    # =========================

    trainer = BpeTrainer(

        vocab_size=VOCAB_SIZE,

        min_frequency=MIN_FREQUENCY,

        special_tokens=[
            "[PAD]",
            "[UNK]",
            "[BOS]",
            "[EOS]"
        ],

        initial_alphabet=ByteLevel.alphabet(),

        show_progress=True
    )


    # =========================
    # 开始训练
    # =========================

    tokenizer.train(
        [
            str(CORPUS)
        ],
        trainer
    )


    # =========================
    # 保存
    # =========================

    tokenizer.save(
        str(SAVE_PATH)
    )


    print("\n========================")
    print("Tokenizer训练完成")
    print("========================")

    print(
        "保存位置:",
        SAVE_PATH
    )

    print(
        "最终词表大小:",
        tokenizer.get_vocab_size()
    )


    # =========================
    # 简单测试
    # =========================

    test_texts = [

        "Carbon nanotube growth by chemical vapor deposition.",

        "Fe nanoparticles catalyze CNT nucleation and growth.",

        "The Raman D and G bands were measured at 1350 and 1580 cm-1.",

        "Vertically aligned carbon nanotube forests were synthesized at 750 °C."
    ]


    print("\n========================")
    print("Tokenizer测试")
    print("========================")


    for text in test_texts:

        encoded = tokenizer.encode(text)

        decoded = tokenizer.decode(encoded.ids)

        print("\n原文:")
        print(text)

        print("\nTokens:")
        print(encoded.tokens)

        print("\nDecode:")
        print(decoded)


if __name__ == "__main__":

    main()