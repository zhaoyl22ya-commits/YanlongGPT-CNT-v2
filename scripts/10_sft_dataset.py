import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizers import Tokenizer


# =====================================================
# 路径
# =====================================================

# scripts/10_sft_dataset.py
BASE_DIR = Path(__file__).resolve().parent.parent


DATA_FILE = (
    BASE_DIR
    /
    "data"
    /
    "sft"
    /
    "cnt_sft_dataset_clean.jsonl"
)


TOKENIZER_FILE = (
    BASE_DIR
    /
    "tokenizer"
    /
    "cnt_tokenizer.json"
)


# =====================================================
# Dataset
# =====================================================

class CNT_SFT_Dataset(Dataset):


    def __init__(
        self,
        max_length=1024
    ):


        self.max_length = max_length


        # =====================================================
        # tokenizer
        # =====================================================

        if not TOKENIZER_FILE.exists():

            raise FileNotFoundError(
                f"找不到Tokenizer文件: {TOKENIZER_FILE}"
            )


        self.tokenizer = Tokenizer.from_file(
            str(TOKENIZER_FILE)
        )


        # =====================================================
        # SFT数据
        # =====================================================

        if not DATA_FILE.exists():

            raise FileNotFoundError(
                f"找不到SFT数据文件: {DATA_FILE}"
            )


        self.samples = []


        print(
            "加载SFT数据..."
        )


        print(
            "数据文件:",
            DATA_FILE
        )


        print(
            "Tokenizer:",
            TOKENIZER_FILE
        )


        print(
            "max_length:",
            self.max_length
        )


        with open(
            DATA_FILE,
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


                    if (
                        "instruction" not in item
                        or
                        "output" not in item
                    ):

                        continue


                    self.samples.append(
                        item
                    )


                except Exception:

                    continue


        print(
            "SFT样本数量:",
            len(self.samples)
        )


        # =====================================================
        # special tokens
        # =====================================================

        self.pad_id = self.tokenizer.token_to_id(
            "[PAD]"
        )


        self.eos_id = self.tokenizer.token_to_id(
            "[EOS]"
        )


        if self.pad_id is None:

            self.pad_id = 0


        if self.eos_id is None:

            raise ValueError(
                "Tokenizer中找不到[EOS] token"
            )


        print(
            "PAD ID:",
            self.pad_id
        )


        print(
            "EOS ID:",
            self.eos_id
        )



    # =====================================================
    # Dataset长度
    # =====================================================

    def __len__(self):

        return len(
            self.samples
        )



    # =====================================================
    # 单个样本
    # =====================================================

    def __getitem__(
        self,
        idx
    ):


        item = self.samples[
            idx
        ]


        instruction = item[
            "instruction"
        ]


        output = item[
            "output"
        ]


        # =====================================================
        # Prompt
        # =====================================================

        prompt = (

            "### Instruction:\n"

            +

            instruction

            +

            "\n\n### Response:\n"

        )


        answer = output


        # =====================================================
        # Tokenize
        # =====================================================

        prompt_ids = self.tokenizer.encode(
            prompt
        ).ids


        answer_ids = self.tokenizer.encode(
            answer
        ).ids


        # 答案最后加入EOS
        answer_ids.append(
            self.eos_id
        )


        # =====================================================
        # GPT next-token prediction
        # =====================================================
        #
        # 非常重要：
        #
        # YanlongGPT.forward()内部不会自动shift。
        #
        # 因此Dataset必须显式构造：
        #
        # input_ids = full_ids[:-1]
        # labels    = full_ids[1:]
        #
        # 不能再次修改这个逻辑。
        # =====================================================

        full_ids = (

            prompt_ids

            +

            answer_ids

        )


        # 输入去掉最后一个token
        input_ids = full_ids[
            :-1
        ]


        # 标签整体向后移动一个token
        labels = full_ids[
            1:
        ]


        # =====================================================
        # Mask Prompt
        # =====================================================
        #
        # Prompt只作为上下文，
        # 不参与loss。
        #
        # 但是：
        # labels[prompt_length - 1]
        # 正好对应answer第一个token，
        # 必须保留。
        # =====================================================

        prompt_length = len(
            prompt_ids
        )


        if prompt_length > 1:

            labels[
                :prompt_length - 1
            ] = (

                [-100]

                *

                (prompt_length - 1)

            )


        # =====================================================
        # Truncate
        # =====================================================

        input_ids = input_ids[
            :self.max_length
        ]


        labels = labels[
            :self.max_length
        ]


        # =====================================================
        # Padding
        # =====================================================

        while len(
            input_ids
        ) < self.max_length:


            input_ids.append(
                self.pad_id
            )


            labels.append(
                -100
            )


        # =====================================================
        # Tensor
        # =====================================================

        return {


            "input_ids":

            torch.tensor(

                input_ids,

                dtype=torch.long

            ),


            "labels":

            torch.tensor(

                labels,

                dtype=torch.long

            )

        }


# =====================================================
# 简单测试
# =====================================================

if __name__ == "__main__":


    dataset = CNT_SFT_Dataset(
        max_length=1024
    )


    print(
        "=============================="
    )


    print(
        "Dataset测试"
    )


    print(
        "=============================="
    )


    print(
        "样本数量:",
        len(dataset)
    )


    if len(dataset) > 0:


        sample = dataset[
            0
        ]


        print(
            "input_ids shape:",
            sample["input_ids"].shape
        )


        print(
            "labels shape:",
            sample["labels"].shape
        )


        valid_labels = (

            sample["labels"]

            !=

            -100

        ).sum().item()


        print(
            "参与loss的token数量:",
            valid_labels
        )


        print(
            "input_ids前20个:"
        )


        print(
            sample["input_ids"][:20]
        )


        print(
            "labels前20个:"
        )


        print(
            sample["labels"][:20]
        )


    print(
        "=============================="
    )


    print(
        "10_sft_dataset.py 测试完成"
    )


    print(
        "=============================="
    )