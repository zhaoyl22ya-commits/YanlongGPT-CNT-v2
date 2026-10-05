import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# =====================================================
# 路径
# =====================================================

THIS_FILE = Path(__file__).resolve()


# 兼容两种放置方式：
#
# scripts/train/11_train_yanlongllm_sft.py
# scripts/11_train_yanlongllm_sft.py

if THIS_FILE.parent.name == "train":

    BASE_DIR = THIS_FILE.parents[2]

else:

    BASE_DIR = THIS_FILE.parent.parent


SCRIPTS_DIR = (
    BASE_DIR
    /
    "scripts"
)


CHECKPOINT_FILE = (
    BASE_DIR
    /
    "checkpoint"
    /
    "yanlongllm_cnt_epoch5.pt"
)


SAVE_DIR = (
    BASE_DIR
    /
    "checkpoint_sft"
)


SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


DATASET_FILE = (
    SCRIPTS_DIR
    /
    "10_sft_dataset.py"
)

# =====================================================
# 导入模型
# =====================================================

MODEL_CODE_FILE = (
    SCRIPTS_DIR
    /
    "06_yanlongllm.py"
)


spec_model = importlib.util.spec_from_file_location(
    "yanlongllm_module",
    MODEL_CODE_FILE
)


yanlongllm_module = importlib.util.module_from_spec(
    spec_model
)


spec_model.loader.exec_module(
    yanlongllm_module
)


YanlongLLM = yanlongllm_module.YanlongLLM
YanlongConfig = yanlongllm_module.YanlongConfig

# =====================================================
# 导入 10_sft_dataset.py
#
# 文件名以数字开头，
# 因此不能直接：
#
# from 10_sft_dataset import ...
#
# 使用 importlib 动态加载。
# =====================================================

if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"找不到 Dataset 文件: {DATASET_FILE}"
    )


spec = importlib.util.spec_from_file_location(
    "sft_dataset_v2",
    DATASET_FILE
)


sft_dataset_module = importlib.util.module_from_spec(
    spec
)


spec.loader.exec_module(
    sft_dataset_module
)


CNT_SFT_Dataset = (
    sft_dataset_module.CNT_SFT_Dataset
)


# =====================================================
# 参数
# =====================================================

# RTX 4060 Laptop 8GB
batch_size = 2


# 梯度累积
gradient_accumulation_steps = 4


# Effective batch size:
# 2 × 4 = 8
effective_batch_size = (
    batch_size
    *
    gradient_accumulation_steps
)


# SFT学习率
learning_rate = 2e-5


# SFT不需要像Base预训练一样跑5轮
epochs = 3


# Windows下保持0最稳定
num_workers = 0


# 随机种子
seed = 42


torch.manual_seed(
    seed
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        seed
    )


# =====================================================
# Device
# =====================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)


use_amp = (
    device
    ==
    "cuda"
)


print(
    "================================="
)

print(
    "YanlongGPT-CNT v2 SFT Training"
)

print(
    "================================="
)


print(
    "device:",
    device
)


print(
    "AMP:",
    use_amp
)


print(
    "batch_size:",
    batch_size
)


print(
    "gradient_accumulation_steps:",
    gradient_accumulation_steps
)


print(
    "effective_batch_size:",
    effective_batch_size
)


print(
    "learning_rate:",
    learning_rate
)


print(
    "epochs:",
    epochs
)


# =====================================================
# 检查Base checkpoint
# =====================================================

if not CHECKPOINT_FILE.exists():

    raise FileNotFoundError(
        f"找不到Base模型Checkpoint: {CHECKPOINT_FILE}"
    )


print(
    "\n加载Base模型:"
)


print(
    CHECKPOINT_FILE
)


# 先加载到CPU，
# 避免直接占GPU产生额外显存峰值
checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu"
)


# =====================================================
# 从Checkpoint读取模型Config
#
# 不再手动写：
#
# vocab_size=10000
# block_size=1024
# n_layer=8
# ...
#
# 避免以后模型参数修改后发生不匹配。
# =====================================================

if "config" not in checkpoint:

    raise KeyError(
        "Base checkpoint中没有找到config"
    )


config_dict = checkpoint[
    "config"
]


config = YanlongConfig(
    **config_dict
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


# =====================================================
# 加载Base模型参数
# =====================================================

if "model" not in checkpoint:

    raise KeyError(
        "Base checkpoint中没有找到model参数"
    )


model.load_state_dict(
    checkpoint["model"]
)


print(
    "\nBase模型加载完成"
)


model.to(
    device
)


parameter_count = sum(

    p.numel()

    for p in model.parameters()

)


print(
    "参数量:",
    parameter_count / 1e6,
    "M"
)


# checkpoint已经加载完成
# 释放CPU中的checkpoint引用
del checkpoint


# =====================================================
# Dataset
#
# max_length直接跟随Base模型block_size。
# v2当前应为1024。
# =====================================================

dataset = CNT_SFT_Dataset(
    max_length=config.block_size
)


if len(dataset) == 0:

    raise ValueError(
        "SFT Dataset为空，无法训练"
    )


# =====================================================
# DataLoader
# =====================================================

train_loader = DataLoader(

    dataset,

    batch_size=batch_size,

    shuffle=True,

    drop_last=False,

    num_workers=num_workers,

    pin_memory=use_amp

)


print(
    "\nSFT样本:",
    len(dataset)
)


print(
    "microsteps / epoch:",
    len(train_loader)
)


estimated_optimizer_steps = (

    len(train_loader)
    +
    gradient_accumulation_steps
    -
    1

) // gradient_accumulation_steps


print(
    "optimizer updates / epoch:",
    estimated_optimizer_steps
)


print(
    "总optimizer updates:",
    estimated_optimizer_steps
    *
    epochs
)


# =====================================================
# Optimizer
# =====================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=learning_rate

)


# =====================================================
# AMP
# =====================================================

scaler = torch.amp.GradScaler(

    "cuda",

    enabled=use_amp

)


# =====================================================
# SFT Training
# =====================================================

print(
    "\n================================="
)

print(
    "开始SFT训练"
)

print(
    "================================="
)


for epoch in range(
    epochs
):


    model.train()


    total_loss = 0.0


    optimizer.zero_grad(
        set_to_none=True
    )


    for step, batch in enumerate(
        train_loader
    ):


        # =================================================
        # 搬到GPU
        # =================================================

        input_ids = batch[
            "input_ids"
        ].to(

            device,

            non_blocking=use_amp

        )


        labels = batch[
            "labels"
        ].to(

            device,

            non_blocking=use_amp

        )


        # =================================================
        # Forward
        # =================================================

        with torch.autocast(

            device_type=device,

            dtype=torch.float16,

            enabled=use_amp

        ):


            logits, loss = model(

                input_ids,

                labels

            )


        # 保存原始loss用于日志
        raw_loss = loss.detach().item()


        total_loss += raw_loss


        # =================================================
        # Gradient Accumulation
        #
        # 每个microbatch只贡献1/4梯度
        # =================================================

        loss = (

            loss

            /

            gradient_accumulation_steps

        )


        scaler.scale(
            loss
        ).backward()


        # =================================================
        # Optimizer Step
        #
        # 每4个microsteps更新一次参数。
        #
        # 最后一批不足4个时也要更新。
        # =================================================

        should_step = (

            (
                (step + 1)
                %
                gradient_accumulation_steps
                ==
                0
            )

            or

            (
                step + 1
                ==
                len(train_loader)
            )

        )


        if should_step:


            scaler.step(
                optimizer
            )


            scaler.update()


            optimizer.zero_grad(
                set_to_none=True
            )


        # =================================================
        # 日志
        # =================================================

        if step % 50 == 0:


            print(

                f"epoch {epoch + 1}/{epochs}",

                f"step {step}/{len(train_loader)}",

                f"loss {raw_loss:.4f}"

            )


    # =====================================================
    # Epoch平均Loss
    # =====================================================

    avg_loss = (

        total_loss

        /

        len(train_loader)

    )


    print(
        "\n===================="
    )


    print(
        "epoch:",
        epoch + 1
    )


    print(
        "SFT loss:",
        avg_loss
    )


    # =====================================================
    # 保存Checkpoint
    # =====================================================

    save_file = (

        SAVE_DIR

        /

        f"yanlongllm_cnt_sft_epoch{epoch + 1}.pt"

    )


    torch.save(

        {

            "model":
                model.state_dict(),

            "config":
                config.__dict__,

            "epoch":
                epoch + 1,

            "sft_loss":
                avg_loss,

            "learning_rate":
                learning_rate,

            "batch_size":
                batch_size,

            "gradient_accumulation_steps":
                gradient_accumulation_steps,

            "effective_batch_size":
                effective_batch_size

        },

        save_file

    )


    print(
        "Checkpoint保存:"
    )


    print(
        save_file
    )


# =====================================================
# 完成
# =====================================================

print(
    "\n================================="
)


print(
    "SFT训练完成"
)


print(
    "模型保存目录:"
)


print(
    SAVE_DIR
)


print(
    "最终模型:"
)


print(

    SAVE_DIR

    /

    f"yanlongllm_cnt_sft_epoch{epochs}.pt"

)


print(
    "================================="
)