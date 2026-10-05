import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


# =====================================================
# 路径
#
# 当前文件：
# 项目根目录/scripts/ragsft/21_rag_sft_dataset.py
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]


DATA_FILE = (
    BASE_DIR
    / "data"
    / "sft"
    / "ragsft"
    / "cnt_rag_sft_dataset_clean.jsonl"
)


TOKENIZER_FILE = (
    BASE_DIR
    / "tokenizer"
    / "cnt_tokenizer.json"
)


# =====================================================
# 模型 / Dataset参数
# =====================================================

DEFAULT_MAX_LENGTH = 1024

TRAIN_RATIO = 0.95
SPLIT_SEED = 42


# =====================================================
# Evidence最小保留量
#
# 极少数样本在真正按照 Evidence 1 / Evidence 2 的格式
# 重新tokenize后可能比20号预估略长。
#
# 如果必须缩短Evidence，则保证每个被选中的chunk
# 至少保留这么多token。
# =====================================================

MIN_EVIDENCE_TOKENS_PER_CHUNK = 16


# =====================================================
# RAG Prompt
#
# 后续22测试和18_chat_rag.py应保持同样格式。
# =====================================================

INSTRUCTION_HEADER = (
    "### Instruction:\n"
    "Use the following retrieved evidence to answer "
    "the CNT question accurately."
)

QUESTION_HEADER = "\n\nQuestion:\n"
RESPONSE_HEADER = "\n\n### Response:\n"


# =====================================================
# 基础检查
# =====================================================

for path, name in [
    (DATA_FILE, "clean RAG-SFT数据"),
    (TOKENIZER_FILE, "Tokenizer"),
]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到{name}: {path}"
        )


# =====================================================
# 工具函数
# =====================================================

def load_clean_items():
    items = []
    failed = 0

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                failed += 1
                continue

            required = [
                "instruction",
                "output",
                "qa_type",
                "source_file",
                "evidence_chunk_ids",
                "evidence_chunks",
                "evidence_text",
            ]

            if not all(
                field in item
                for field in required
            ):
                failed += 1
                continue

            question = str(
                item.get("instruction", "")
            ).strip()

            answer = str(
                item.get("output", "")
            ).strip()

            source_file = str(
                item.get("source_file", "")
            ).strip()

            evidence_chunks = item.get(
                "evidence_chunks",
                [],
            )

            if (
                not question
                or not answer
                or not source_file
                or not isinstance(evidence_chunks, list)
                or len(evidence_chunks) == 0
            ):
                failed += 1
                continue

            items.append(item)

    return items, failed


def build_paper_split(
    items,
    train_ratio=TRAIN_RATIO,
    seed=SPLIT_SEED,
):
    """
    按source_file做paper-level split。

    同一篇论文产生的所有QA只能进入同一个split，
    避免同源论文在train/val之间泄漏。
    """

    paper_names = sorted(
        {
            str(item["source_file"]).strip()
            for item in items
        }
    )

    if len(paper_names) < 2:
        return set(paper_names), set()

    rng = random.Random(seed)
    rng.shuffle(paper_names)

    val_count = int(
        round(
            len(paper_names)
            * (1.0 - train_ratio)
        )
    )

    val_count = max(
        1,
        min(
            val_count,
            len(paper_names) - 1,
        ),
    )

    val_papers = set(
        paper_names[:val_count]
    )

    train_papers = set(
        paper_names[val_count:]
    )

    return train_papers, val_papers


def build_evidence_header(index):
    return f"\n\nEvidence {index}:\n"


# =====================================================
# Dataset
# =====================================================

class CNT_RAG_SFT_Dataset(Dataset):

    def __init__(
        self,
        split="all",
        max_length=DEFAULT_MAX_LENGTH,
        train_ratio=TRAIN_RATIO,
        split_seed=SPLIT_SEED,
        verbose=True,
    ):
        """
        split:
            "all"   -> 全部clean RAG-SFT
            "train" -> paper-level train split
            "val"   -> paper-level validation split
        """

        if split not in {
            "all",
            "train",
            "val",
        }:
            raise ValueError(
                "split必须是 all / train / val"
            )

        self.split = split
        self.max_length = int(max_length)
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.verbose = bool(verbose)

        if self.max_length < 64:
            raise ValueError(
                "max_length过小"
            )

        # =================================================
        # Tokenizer
        # =================================================

        self.tokenizer = Tokenizer.from_file(
            str(TOKENIZER_FILE)
        )

        self.pad_id = self.tokenizer.token_to_id(
            "[PAD]"
        )

        self.eos_id = self.tokenizer.token_to_id(
            "[EOS]"
        )

        if self.pad_id is None:
            self.pad_id = 0

        # =================================================
        # 数据
        # =================================================

        all_items, load_failed = load_clean_items()

        train_papers, val_papers = build_paper_split(
            all_items,
            train_ratio=self.train_ratio,
            seed=self.split_seed,
        )

        if split == "all":
            selected_items = all_items

        elif split == "train":
            selected_items = [
                item
                for item in all_items
                if item["source_file"]
                in train_papers
            ]

        else:
            selected_items = [
                item
                for item in all_items
                if item["source_file"]
                in val_papers
            ]

        self.samples = selected_items

        self.train_papers = train_papers
        self.val_papers = val_papers

        self.stats = {
            "all_samples": len(all_items),
            "selected_samples": len(self.samples),
            "load_failed": load_failed,
            "all_papers": len(
                {
                    item["source_file"]
                    for item in all_items
                }
            ),
            "train_papers": len(train_papers),
            "val_papers": len(val_papers),
        }

        if self.verbose:
            self._print_summary()

    # =====================================================
    # 基础信息
    # =====================================================

    def _print_summary(self):
        print(
            "================================="
        )
        print(
            "YanlongGPT-CNT RAG-SFT Dataset"
        )
        print(
            "================================="
        )

        print(
            "数据:"
        )
        print(
            DATA_FILE
        )

        print(
            "Tokenizer:"
        )
        print(
            TOKENIZER_FILE
        )

        print(
            "split:",
            self.split,
        )

        print(
            "max_length:",
            self.max_length,
        )

        print(
            "paper split seed:",
            self.split_seed,
        )

        print(
            "train ratio:",
            self.train_ratio,
        )

        print(
            "全部clean样本:",
            self.stats["all_samples"],
        )

        print(
            "读取失败:",
            self.stats["load_failed"],
        )

        print(
            "全部来源论文:",
            self.stats["all_papers"],
        )

        print(
            "train论文:",
            self.stats["train_papers"],
        )

        print(
            "val论文:",
            self.stats["val_papers"],
        )

        print(
            "当前split样本:",
            self.stats["selected_samples"],
        )

        print(
            "================================="
        )

    # =====================================================
    # 长度
    # =====================================================

    def __len__(self):
        return len(self.samples)

    # =====================================================
    # Evidence读取
    # =====================================================

    def _get_evidence_texts(self, item):
        evidence_chunks = item.get(
            "evidence_chunks",
            [],
        )

        texts = []

        for chunk in evidence_chunks:
            if not isinstance(chunk, dict):
                continue

            text = str(
                chunk.get("text", "")
            ).strip()

            if text:
                texts.append(text)

        if not texts:
            raise RuntimeError(
                "样本缺少有效Evidence文本"
            )

        return texts

    # =====================================================
    # Prompt构建
    # =====================================================

    def _build_prompt_text(
        self,
        question,
        evidence_texts,
    ):
        parts = [
            INSTRUCTION_HEADER
        ]

        for index, evidence in enumerate(
            evidence_texts,
            start=1,
        ):
            parts.append(
                build_evidence_header(index)
            )
            parts.append(evidence)

        parts.append(
            QUESTION_HEADER
        )
        parts.append(question)
        parts.append(
            RESPONSE_HEADER
        )

        return "".join(parts)

    # =====================================================
    # 给Evidence token预算做公平分配
    #
    # 如果只有极少数样本超过1024：
    # 不截Question，不截Answer，
    # 只压缩Evidence。
    #
    # 对2个chunk时不会简单地把第二个chunk整个砍掉，
    # 而是按各chunk长度比例分配预算。
    # =====================================================

    def _allocate_evidence_tokens(
        self,
        evidence_token_lists,
        total_budget,
    ):
        n = len(
            evidence_token_lists
        )

        if n == 0:
            return []

        lengths = [
            len(tokens)
            for tokens in evidence_token_lists
        ]

        if sum(lengths) <= total_budget:
            return lengths

        minimums = [
            min(
                MIN_EVIDENCE_TOKENS_PER_CHUNK,
                length,
            )
            for length in lengths
        ]

        min_total = sum(minimums)

        if min_total > total_budget:
            raise RuntimeError(
                "Evidence token预算过小，无法为每个chunk保留最小上下文"
            )

        allocations = list(minimums)

        remaining = (
            total_budget
            - min_total
        )

        capacities = [
            length - minimum
            for length, minimum in zip(
                lengths,
                minimums,
            )
        ]

        total_capacity = sum(capacities)

        if (
            remaining > 0
            and total_capacity > 0
        ):
            # 先按容量比例分配
            proportional = []

            for capacity in capacities:
                share = int(
                    remaining
                    * capacity
                    / total_capacity
                )

                proportional.append(
                    min(
                        share,
                        capacity,
                    )
                )

            for i, share in enumerate(
                proportional
            ):
                allocations[i] += share

            used = sum(allocations)
            leftover = total_budget - used

            # 处理整数取整产生的剩余token
            while leftover > 0:
                changed = False

                for i in range(n):
                    if allocations[i] < lengths[i]:
                        allocations[i] += 1
                        leftover -= 1
                        changed = True

                        if leftover == 0:
                            break

                if not changed:
                    break

        return allocations

    # =====================================================
    # 构造真正Prompt IDs
    # =====================================================

    def _build_prompt_ids(
        self,
        question,
        evidence_texts,
        answer_ids,
    ):
        full_token_budget = (
            self.max_length
            + 1
        )

        # -------------------------------------------------
        # 先尝试完整Evidence
        # -------------------------------------------------

        full_prompt_text = self._build_prompt_text(
            question,
            evidence_texts,
        )

        full_prompt_ids = self.tokenizer.encode(
            full_prompt_text
        ).ids

        if (
            len(full_prompt_ids)
            + len(answer_ids)
            <= full_token_budget
        ):
            return (
                full_prompt_ids,
                False,
            )

        # -------------------------------------------------
        # 需要截Evidence
        #
        # 用空Evidence构造固定开销，
        # 再将剩余预算公平分给各chunk。
        # -------------------------------------------------

        empty_evidence_texts = [
            ""
            for _ in evidence_texts
        ]

        base_prompt_text = self._build_prompt_text(
            question,
            empty_evidence_texts,
        )

        base_prompt_ids = self.tokenizer.encode(
            base_prompt_text
        ).ids

        evidence_budget = (
            full_token_budget
            - len(base_prompt_ids)
            - len(answer_ids)
        )

        minimum_required = sum(
            min(
                MIN_EVIDENCE_TOKENS_PER_CHUNK,
                len(
                    self.tokenizer.encode(text).ids
                ),
            )
            for text in evidence_texts
        )

        if evidence_budget < minimum_required:
            raise RuntimeError(
                "Question + Answer过长，无法保留必要Evidence"
            )

        evidence_token_lists = [
            self.tokenizer.encode(text).ids
            for text in evidence_texts
        ]

        allocations = self._allocate_evidence_tokens(
            evidence_token_lists,
            evidence_budget,
        )

        # -------------------------------------------------
        # Token截断后decode回文本，再按完整Prompt统一encode。
        # -------------------------------------------------

        def rebuild_prompt(current_allocations):
            truncated_texts = []

            for tokens, allocation in zip(
                evidence_token_lists,
                current_allocations,
            ):
                kept = tokens[:allocation]

                text = self.tokenizer.decode(
                    kept,
                    skip_special_tokens=False,
                ).strip()

                truncated_texts.append(text)

            prompt_text = self._build_prompt_text(
                question,
                truncated_texts,
            )

            prompt_ids = self.tokenizer.encode(
                prompt_text
            ).ids

            return prompt_ids

        prompt_ids = rebuild_prompt(
            allocations
        )

        # -------------------------------------------------
        # 因为“分别tokenize后再decode”和“整体tokenize”
        # 在边界处可能有极小token差异，
        # 所以再做一个最终安全收缩。
        # -------------------------------------------------

        while (
            len(prompt_ids)
            + len(answer_ids)
            > full_token_budget
        ):
            reducible = [
                allocations[i]
                - min(
                    MIN_EVIDENCE_TOKENS_PER_CHUNK,
                    len(evidence_token_lists[i]),
                )
                for i in range(
                    len(allocations)
                )
            ]

            max_reducible = max(reducible)

            if max_reducible <= 0:
                raise RuntimeError(
                    "Evidence已经压缩到最小仍超过1024 token"
                )

            index = reducible.index(
                max_reducible
            )

            overflow = (
                len(prompt_ids)
                + len(answer_ids)
                - full_token_budget
            )

            reduction = min(
                max(
                    1,
                    overflow + 2,
                ),
                max_reducible,
            )

            allocations[index] -= reduction

            prompt_ids = rebuild_prompt(
                allocations
            )

        return (
            prompt_ids,
            True,
        )

    # =====================================================
    # 单条数据编码
    # =====================================================

    def encode_item(
        self,
        item,
    ):
        question = str(
            item["instruction"]
        ).strip()

        answer = str(
            item["output"]
        ).strip()

        evidence_texts = self._get_evidence_texts(
            item
        )

        # =================================================
        # Answer
        # =================================================

        answer_ids = self.tokenizer.encode(
            answer
        ).ids

        if self.eos_id is not None:
            answer_ids = (
                answer_ids
                + [self.eos_id]
            )

        if not answer_ids:
            raise RuntimeError(
                "Answer tokenize后为空"
            )

        # =================================================
        # Prompt
        # =================================================

        prompt_ids, evidence_truncated = (
            self._build_prompt_ids(
                question=question,
                evidence_texts=evidence_texts,
                answer_ids=answer_ids,
            )
        )

        prompt_length = len(
            prompt_ids
        )

        # =================================================
        # 显式next-token shift
        #
        # YanlongLLM.forward()内部不做shift：
        #
        # input  = full[:-1]
        # label  = full[1:]
        # =================================================

        full_ids = (
            prompt_ids
            + answer_ids
        )

        input_ids = full_ids[:-1]
        labels = full_ids[1:]

        # =================================================
        # Answer-only loss
        #
        # labels[prompt_length - 1]
        # 正好对应Answer第一个token，必须保留。
        # =================================================

        if prompt_length > 1:
            labels[
                :prompt_length - 1
            ] = [
                -100
            ] * (
                prompt_length - 1
            )

        # =================================================
        # 长度安全
        # =================================================

        if len(input_ids) > self.max_length:
            raise RuntimeError(
                "RAG-SFT长度错误: "
                f"{len(input_ids)} > {self.max_length}"
            )

        if len(labels) != len(input_ids):
            raise RuntimeError(
                "input_ids / labels长度不一致"
            )

        # =================================================
        # Padding
        # =================================================

        padding_length = (
            self.max_length
            - len(input_ids)
        )

        if padding_length > 0:
            input_ids.extend(
                [self.pad_id]
                * padding_length
            )

            labels.extend(
                [-100]
                * padding_length
            )

        # =================================================
        # Tensor
        # =================================================

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
            "evidence_truncated": evidence_truncated,
        }

    # =====================================================
    # PyTorch Dataset接口
    #
    # 训练时只返回input_ids / labels，
    # 不把metadata交给模型。
    # =====================================================

    def __getitem__(
        self,
        idx,
    ):
        encoded = self.encode_item(
            self.samples[idx]
        )

        return {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
        }


# =====================================================
# 完整自检
# =====================================================

def validate_dataset(
    dataset,
):
    truncated_count = 0
    supervised_token_counts = []
    qa_type_counter = Counter()
    domain_counter = Counter()
    evidence_chunk_counter = Counter()

    for index in range(
        len(dataset)
    ):
        item = dataset.samples[index]

        encoded = dataset.encode_item(
            item
        )

        input_ids = encoded[
            "input_ids"
        ]

        labels = encoded[
            "labels"
        ]

        if input_ids.shape[0] != dataset.max_length:
            raise RuntimeError(
                f"样本{index} input_ids长度异常"
            )

        if labels.shape[0] != dataset.max_length:
            raise RuntimeError(
                f"样本{index} labels长度异常"
            )

        supervised_tokens = int(
            (
                labels != -100
            ).sum().item()
        )

        if supervised_tokens <= 0:
            raise RuntimeError(
                f"样本{index}没有监督Answer token"
            )

        supervised_token_counts.append(
            supervised_tokens
        )

        if encoded[
            "evidence_truncated"
        ]:
            truncated_count += 1

        qa_type_counter[
            item.get(
                "qa_type",
                "unknown",
            )
        ] += 1

        domain_counter[
            item.get(
                "domain_scope",
                "unknown",
            )
        ] += 1

        evidence_chunk_counter[
            len(
                item.get(
                    "evidence_chunk_ids",
                    [],
                )
            )
        ] += 1

    return {
        "truncated_count": truncated_count,
        "supervised_token_counts": supervised_token_counts,
        "qa_type_counter": qa_type_counter,
        "domain_counter": domain_counter,
        "evidence_chunk_counter": evidence_chunk_counter,
    }


def print_basic_stats(
    title,
    values,
):
    if not values:
        print(
            f"{title}: 无数据"
        )
        return

    values = sorted(values)
    n = len(values)

    mean_value = sum(values) / n
    p50 = values[
        int(
            0.50 * (n - 1)
        )
    ]
    p95 = values[
        int(
            0.95 * (n - 1)
        )
    ]

    print(
        f"{title}: "
        f"mean={mean_value:.1f}, "
        f"p50={p50}, "
        f"p95={p95}, "
        f"max={values[-1]}"
    )


# =====================================================
# main
# =====================================================

def main():
    print(
        "================================="
    )
    print(
        "21 RAG-SFT Dataset Full Check"
    )
    print(
        "================================="
    )

    # =================================================
    # ALL
    # =================================================

    all_dataset = CNT_RAG_SFT_Dataset(
        split="all",
        max_length=DEFAULT_MAX_LENGTH,
        verbose=True,
    )

    if len(all_dataset) == 0:
        raise RuntimeError(
            "RAG-SFT Dataset为空"
        )

    print(
        "\n开始逐条检查全部样本..."
    )

    check = validate_dataset(
        all_dataset
    )

    print(
        "全部样本Tensor检查通过: ",
        len(all_dataset),
    )

    print(
        "真正需要Evidence截断的样本:",
        check["truncated_count"],
    )

    print_basic_stats(
        "Supervised answer tokens",
        check[
            "supervised_token_counts"
        ],
    )

    print(
        "\nQA类型分布:"
    )

    for key, value in sorted(
        check[
            "qa_type_counter"
        ].items()
    ):
        print(
            f"- {key}: {value}"
        )

    print(
        "\n领域范围分布:"
    )

    for key, value in sorted(
        check[
            "domain_counter"
        ].items()
    ):
        print(
            f"- {key}: {value}"
        )

    print(
        "\nEvidence chunk数量:"
    )

    for key, value in sorted(
        check[
            "evidence_chunk_counter"
        ].items()
    ):
        print(
            f"- {key} chunk: {value}"
        )

    # =================================================
    # Train / Val paper-level split
    # =================================================

    train_dataset = CNT_RAG_SFT_Dataset(
        split="train",
        max_length=DEFAULT_MAX_LENGTH,
        verbose=False,
    )

    val_dataset = CNT_RAG_SFT_Dataset(
        split="val",
        max_length=DEFAULT_MAX_LENGTH,
        verbose=False,
    )

    train_sources = {
        item["source_file"]
        for item in train_dataset.samples
    }

    val_sources = {
        item["source_file"]
        for item in val_dataset.samples
    }

    overlap = (
        train_sources
        & val_sources
    )

    if overlap:
        raise RuntimeError(
            "paper-level split泄漏："
            f"{len(overlap)}篇论文同时出现在train和val"
        )

    if (
        len(train_dataset)
        + len(val_dataset)
        != len(all_dataset)
    ):
        raise RuntimeError(
            "train + val样本数不等于all"
        )

    print(
        "\n================================="
    )
    print(
        "Paper-level Split"
    )
    print(
        "================================="
    )

    print(
        "Train samples:",
        len(train_dataset),
    )

    print(
        "Val samples:",
        len(val_dataset),
    )

    print(
        "Train papers:",
        len(train_sources),
    )

    print(
        "Val papers:",
        len(val_sources),
    )

    print(
        "Train/Val paper overlap:",
        len(overlap),
    )

    # =================================================
    # 第一条样本Tensor
    # =================================================

    first_sample = train_dataset[0]

    print(
        "\n================================="
    )
    print(
        "Tensor检查"
    )
    print(
        "================================="
    )

    print(
        "input_ids shape:",
        tuple(
            first_sample[
                "input_ids"
            ].shape
        ),
    )

    print(
        "labels shape:",
        tuple(
            first_sample[
                "labels"
            ].shape
        ),
    )

    supervised_tokens = int(
        (
            first_sample[
                "labels"
            ]
            != -100
        ).sum().item()
    )

    print(
        "supervised answer tokens:",
        supervised_tokens,
    )

    print(
        "\n================================="
    )
    print(
        "21_rag_sft_dataset.py 检查完成"
    )
    print(
        "Dataset可以用于RAG-aware SFT训练"
    )
    print(
        "下一步: train/train_yanlongllm_ragsft.py"
    )
    print(
        "================================="
    )


if __name__ == "__main__":
    main()
