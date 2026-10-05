import torch
from pathlib import Path
import importlib.util

from tokenizers import Tokenizer


# =====================================================
# 路径
# =====================================================

# scripts/12_chat_sft.py
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
    "YanlongGPT-CNT v2 SFT Chat"
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


eos_id = tokenizer.token_to_id(
    "[EOS]"
)


print(
    "vocab:",
    tokenizer.get_vocab_size()
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
# 从checkpoint读取config
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
# model
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
    "\n参数量:",
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


del checkpoint


print(
    "\n================================="
)

print(
    "YanlongGPT-CNT SFT 已加载"
)

print(
    "输入 exit / quit / q 退出"
)

print(
    "================================="
)


# =====================================================
# generate
# =====================================================

@torch.no_grad()
def generate(
    question,
    max_new_tokens=256,
    temperature=0.3,
    top_k=10,
    repetition_penalty=1.2
):


    # =================================================
    # 必须和SFT训练格式保持一致
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
        #
        # 只惩罚已经生成的回答token，
        # 不惩罚问题Prompt中的CNT等专业词。
        # =================================================

        generated_part = input_ids[
            0,
            len(prompt_ids):
        ].tolist()


        generated_tokens = set(
            generated_part
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
        # top-k sampling
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
    # 去掉Prompt，只保留回答
    # =====================================================

    output_ids = (

        input_ids[0]

        .tolist()

        [
            len(prompt_ids):
        ]

    )


    # 删除结尾EOS
    if (
        eos_id is not None
        and
        len(output_ids) > 0
        and
        output_ids[-1] == eos_id
    ):

        output_ids = output_ids[:-1]


    answer = tokenizer.decode(
        output_ids
    )


    return answer.strip()


# =====================================================
# chat loop
# =====================================================

if __name__ == "__main__":


    while True:


        print(
            "\nUser:"
        )


        question = input(
            "> "
        ).strip()


        if question.lower() in [

            "exit",

            "quit",

            "q"

        ]:

            print(
                "Bye."
            )

            break


        if not question:

            continue


        answer = generate(

            question,

            max_new_tokens=256,

            temperature=0.3,

            top_k=10,

            repetition_penalty=1.2

        )


        print(
            "\nYanlongGPT:"
        )


        print(
            answer
        )