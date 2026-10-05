import torch
from pathlib import Path
import importlib.util

from tokenizers import Tokenizer


# =====================================================
# 路径
# =====================================================

# scripts/11_test_sft.py
BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_FILE = (
    BASE_DIR
    /
    "checkpoint_sft"
    /
    "yanlongllm_cnt_sft_epoch3.pt"
)


TOKENIZER_FILE = (
    BASE_DIR
    /
    "tokenizer"
    /
    "cnt_tokenizer.json"
)

# =====================================================
# import
# =====================================================

MODEL_CODE_FILE = (
    BASE_DIR
    /
    "scripts"
    /
    "06_yanlongllm.py"
)


spec = importlib.util.spec_from_file_location(
    "yanlongllm_module",
    MODEL_CODE_FILE
)


yanlongllm_module = importlib.util.module_from_spec(
    spec
)


spec.loader.exec_module(
    yanlongllm_module
)


YanlongLLM = yanlongllm_module.YanlongLLM
YanlongConfig = yanlongllm_module.YanlongConfig

# =====================================================
# device
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "================================="
)

print(
    "YanlongGPT-CNT v2 SFT Test"
)

print(
    "================================="
)


print(
    "device:",
    device
)


# =====================================================
# 检查文件
# =====================================================

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"找不到SFT模型: {MODEL_FILE}"
    )


if not TOKENIZER_FILE.exists():

    raise FileNotFoundError(
        f"找不到Tokenizer: {TOKENIZER_FILE}"
    )


# =====================================================
# tokenizer
# =====================================================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_FILE)
)


print(
    "词表大小:",
    tokenizer.get_vocab_size()
)


eos_id = tokenizer.token_to_id(
    "[EOS]"
)


print(
    "EOS:",
    eos_id
)


# =====================================================
# 加载checkpoint
# =====================================================

print(
    "\n加载SFT模型:"
)


print(
    MODEL_FILE
)


checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu"
)


# =====================================================
# 从checkpoint读取模型config
# =====================================================

if "config" not in checkpoint:

    raise KeyError(
        "checkpoint中没有找到config"
    )


config = YanlongConfig(
    **checkpoint["config"]
)


print(
    "\n模型Config:"
)


print(
    "vocab_size:",
    config.vocab_size
)


print(
    "block_size:",
    config.block_size
)


print(
    "n_layer:",
    config.n_layer
)


print(
    "n_head:",
    config.n_head
)


print(
    "n_embd:",
    config.n_embd
)


# =====================================================
# 创建模型
# =====================================================

model = YanlongLLM(
    config
)


if "model" not in checkpoint:

    raise KeyError(
        "checkpoint中没有找到model参数"
    )


model.load_state_dict(
    checkpoint["model"]
)


model.to(
    device
)


model.eval()


parameter_count = sum(

    p.numel()

    for p in model.parameters()

)


print(
    "\nSFT模型加载完成"
)


print(
    "参数量:",
    parameter_count / 1e6,
    "M"
)


if "epoch" in checkpoint:

    print(
        "SFT epoch:",
        checkpoint["epoch"]
    )


if "sft_loss" in checkpoint:

    print(
        "SFT loss:",
        checkpoint["sft_loss"]
    )


# checkpoint加载完成后可释放
del checkpoint


# =====================================================
# generate
# =====================================================

@torch.no_grad()
def generate(
    question,
    max_new_tokens=200,
    temperature=0.3,
    top_k=10,
    repetition_penalty=1.2
):


    # =================================================
    # 必须与SFT训练Prompt完全一致
    # =================================================

    prompt = (

        "### Instruction:\n"

        +

        question

        +

        "\n\n### Response:\n"

    )


    prompt_ids = tokenizer.encode(
        prompt
    ).ids


    input_ids = torch.tensor(

        prompt_ids,

        dtype=torch.long

    ).unsqueeze(0)


    input_ids = input_ids.to(
        device
    )


    # =================================================
    # 自回归生成
    # =================================================

    for _ in range(
        max_new_tokens
    ):


        # 使用模型真实block_size
        idx = input_ids[
            :,
            -config.block_size:
        ]


        logits, _ = model(
            idx
        )


        logits = logits[
            :,
            -1,
            :
        ]


        # =================================================
        # repetition penalty
        # =================================================

        generated_tokens = set(
            input_ids[0].tolist()
        )


        for token_id in generated_tokens:


            if logits[
                0,
                token_id
            ] > 0:

                logits[
                    0,
                    token_id
                ] /= repetition_penalty

            else:

                logits[
                    0,
                    token_id
                ] *= repetition_penalty


        # =================================================
        # top-k
        # =================================================

        k = min(
            top_k,
            logits.size(-1)
        )


        values, indices = torch.topk(

            logits,

            k

        )


        # =================================================
        # temperature
        # =================================================

        values = (
            values
            /
            temperature
        )


        probs = torch.softmax(

            values,

            dim=-1

        )


        choice = torch.multinomial(

            probs,

            num_samples=1

        )


        next_token = indices.gather(

            -1,

            choice

        )


        input_ids = torch.cat(

            [
                input_ids,
                next_token
            ],

            dim=1

        )


        # =================================================
        # EOS停止
        # =================================================

        if eos_id is not None:


            if next_token.item() == eos_id:

                break


    # =====================================================
    # 去掉Prompt，只保留模型答案
    # =====================================================

    generated_ids = (

        input_ids[0]

        .tolist()

        [
            len(prompt_ids):
        ]

    )


    # 如果最后有EOS，移除
    if (
        eos_id is not None
        and
        len(generated_ids) > 0
        and
        generated_ids[-1] == eos_id
    ):

        generated_ids = (
            generated_ids[:-1]
        )


    answer = tokenizer.decode(
        generated_ids
    )


    return answer.strip()


# =====================================================
# 测试问题
# =====================================================

if __name__ == "__main__":


    questions = [


        "How does catalyst particle size affect carbon nanotube growth?",


        "How does temperature influence CNT structure and yield during CVD growth?",


        "Why are Fe catalysts commonly used for carbon nanotube growth?",


        "How can the G peak and D peak in Raman spectroscopy be used to evaluate CNT quality?",


        "How can CNT diameter be controlled through experimental conditions?"

    ]


    print(
        "\n================================="
    )

    print(
        "开始SFT问答测试"
    )

    print(
        "================================="
    )


    for q in questions:


        print(
            "\n=============================="
        )


        print(
            "Question:"
        )


        print(
            q
        )


        print(
            "\nAnswer:"
        )


        answer = generate(

            q,

            max_new_tokens=200,

            temperature=0.3,

            top_k=10,

            repetition_penalty=1.2

        )


        print(
            answer
        )


    print(
        "\n================================="
    )

    print(
        "11_test_sft.py 测试完成"
    )

    print(
        "================================="
    )