import torch
from pathlib import Path
from tokenizers import Tokenizer
import importlib.util


# =========================
# 路径
# =========================

# 当前文件位于：
# 项目根目录/scripts/07_inference_yanlongllm.py
BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    BASE_DIR
    / "checkpoint"
    / "yanlongllm_cnt_epoch5.pt"
)


TOKENIZER_PATH = (
    BASE_DIR
    / "tokenizer"
    / "cnt_tokenizer.json"
)

# =========================
# 导入模型
# =========================

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

# =========================
# 设备
# =========================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("device:", device)


# =========================
# tokenizer
# =========================

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)


print(
    "词表大小:",
    tokenizer.get_vocab_size()
)


bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")


# =========================
# 加载 checkpoint
# =========================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# 直接读取训练时保存的模型配置
config = YanlongConfig(
    **checkpoint["config"]
)


print("===================")

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


# =========================
# 创建并加载模型
# =========================

model = YanlongLLM(
    config
)


model.load_state_dict(
    checkpoint["model"]
)


model.to(device)

model.eval()


print("模型加载完成")


# =========================
# 生成函数
# =========================

@torch.no_grad()
def generate(
        prompt,
        max_new_tokens=200,
        temperature=0.8
):


    tokens = tokenizer.encode(
        prompt
    ).ids


    # 与预训练格式保持一致
    if bos_id is not None:

        tokens = [
            bos_id
        ] + tokens


    x = torch.tensor(
        tokens,
        dtype=torch.long
    ).unsqueeze(0).to(device)



    for _ in range(max_new_tokens):


        # 超过最大上下文长度时截断
        x_cond = x[
            :,
            -config.block_size:
        ]


        logits, _ = model(
            x_cond
        )


        logits = logits[
            :,
            -1,
            :
        ]


        # temperature
        logits = (
            logits
            /
            temperature
        )


        probs = torch.softmax(
            logits,
            dim=-1
        )


        next_token = torch.multinomial(
            probs,
            num_samples=1
        )


        # 如果生成 EOS，停止
        if (
            eos_id is not None
            and
            next_token.item() == eos_id
        ):

            break


        x = torch.cat(
            [
                x,
                next_token
            ],
            dim=1
        )



    output_ids = x[0].tolist()


    # 不显示开头 BOS
    if (
        bos_id is not None
        and
        len(output_ids) > 0
        and
        output_ids[0] == bos_id
    ):

        output_ids = output_ids[1:]


    text = tokenizer.decode(
        output_ids
    )


    return text



# =========================
# 测试
# =========================

prompt = (
    "Carbon nanotube growth by chemical vapor deposition"
)


result = generate(
    prompt,
    max_new_tokens=200,
    temperature=0.8
)


print("===================")

print("输入:")

print(prompt)


print("===================")

print("模型输出:")

print(result)