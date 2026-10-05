import sys
import math
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# =====================================================
# 路径
#
# 当前文件：
# 项目根目录/scripts/train/train_yanlongllm_ragsft.py
# =====================================================

THIS_FILE = Path(__file__).resolve()

if THIS_FILE.parent.name == "train":
    BASE_DIR = THIS_FILE.parents[2]
else:
    BASE_DIR = THIS_FILE.parent.parent

SCRIPTS_DIR = (
    BASE_DIR
    / "scripts"
)

MODEL_FILE = (
    SCRIPTS_DIR
    / "06_yanlongllm.py"
)

DATASET_FILE = (
    SCRIPTS_DIR
    / "ragsft"
    / "21_rag_sft_dataset.py"
)

# =====================================================
# 起点：
# 普通SFT已经训练完成的epoch3模型
# =====================================================

CHECKPOINT_FILE = (
    BASE_DIR
    / "checkpoint_sft"
    / "yanlongllm_cnt_sft_epoch3.pt"
)

# =====================================================
# RAG-aware SFT输出目录
# =====================================================

SAVE_DIR = (
    BASE_DIR
    / "checkpoint_ragsft"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT_FILE = (
    SAVE_DIR
    / "yanlongllm_cnt_ragsft_best.pt"
)


# =====================================================
# 参数
#
# RTX 4060 Laptop 8GB
# =====================================================

batch_size = 2

gradient_accumulation_steps = 4

effective_batch_size = (
    batch_size
    *
    gradient_accumulation_steps
)

# RAG-aware SFT是在普通SFT epoch3上继续训练，
# 学习率进一步降低。
learning_rate = 1e-5

# 先训练2轮。
# 数据已经是高质量、强约束的Grounded QA，
# 不建议一开始跑很多epoch。
epochs = 2

# Windows下0最稳定
num_workers = 0

# 与21号paper-level split保持一致
seed = 42

# 梯度裁剪
max_grad_norm = 1.0

# 日志间隔
log_interval = 50


# =====================================================
# 随机种子
# =====================================================

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


# =====================================================
# 动态加载 06_yanlongllm.py
#
# 文件名以数字开头，不能普通import。
# =====================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"找不到模型代码: {MODEL_FILE}"
    )

model_spec = importlib.util.spec_from_file_location(
    "yanlongllm_module",
    MODEL_FILE
)

yanlongllm_module = importlib.util.module_from_spec(
    model_spec
)

sys.modules[
    model_spec.name
] = yanlongllm_module

model_spec.loader.exec_module(
    yanlongllm_module
)

YanlongLLM = (
    yanlongllm_module.YanlongLLM
)

YanlongConfig = (
    yanlongllm_module.YanlongConfig
)


# =====================================================
# 动态加载 21_rag_sft_dataset.py
# =====================================================

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"找不到RAG-SFT Dataset代码: {DATASET_FILE}"
    )

dataset_spec = importlib.util.spec_from_file_location(
    "rag_sft_dataset_module",
    DATASET_FILE
)

rag_sft_dataset_module = (
    importlib.util.module_from_spec(
        dataset_spec
    )
)

sys.modules[
    dataset_spec.name
] = rag_sft_dataset_module

dataset_spec.loader.exec_module(
    rag_sft_dataset_module
)

CNT_RAG_SFT_Dataset = (
    rag_sft_dataset_module.CNT_RAG_SFT_Dataset
)


# =====================================================
# 启动信息
# =====================================================

print(
    "================================="
)

print(
    "YanlongGPT-CNT v2 RAG-aware SFT Training"
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

print(
    "max_grad_norm:",
    max_grad_norm
)

print(
    "seed:",
    seed
)


# =====================================================
# 检查Checkpoint
# =====================================================

if not CHECKPOINT_FILE.exists():
    raise FileNotFoundError(
        f"找不到普通SFT epoch3 checkpoint: {CHECKPOINT_FILE}"
    )

print(
    "\n加载普通SFT模型:"
)

print(
    CHECKPOINT_FILE
)


# =====================================================
# 加载Checkpoint到CPU
# =====================================================

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location="cpu"
)

if "model" not in checkpoint:
    raise KeyError(
        "Checkpoint中没有找到model"
    )

if "config" not in checkpoint:
    raise KeyError(
        "Checkpoint中没有找到config"
    )


# =====================================================
# Model config
# =====================================================

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
# 创建模型并加载普通SFT参数
# =====================================================

model = YanlongLLM(
    config
)

model.load_state_dict(
    checkpoint[
        "model"
    ]
)

print(
    "\n普通SFT epoch3模型加载完成"
)

model.to(
    device
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "总参数量:",
    parameter_count / 1e6,
    "M"
)

print(
    "可训练参数量:",
    trainable_parameter_count / 1e6,
    "M"
)

# 释放CPU checkpoint引用
del checkpoint


# =====================================================
# Dataset
#
# 使用21号中的paper-level 95/5 split。
#
# max_length必须跟随模型block_size。
# =====================================================

train_dataset = CNT_RAG_SFT_Dataset(
    split="train",
    max_length=config.block_size,
    train_ratio=0.95,
    split_seed=seed,
    verbose=True,
)

val_dataset = CNT_RAG_SFT_Dataset(
    split="val",
    max_length=config.block_size,
    train_ratio=0.95,
    split_seed=seed,
    verbose=True,
)

if len(train_dataset) == 0:
    raise ValueError(
        "RAG-SFT train Dataset为空"
    )

if len(val_dataset) == 0:
    raise ValueError(
        "RAG-SFT val Dataset为空"
    )


# =====================================================
# 二次检查paper-level无泄漏
# =====================================================

train_sources = {
    item["source_file"]
    for item in train_dataset.samples
}

val_sources = {
    item["source_file"]
    for item in val_dataset.samples
}

paper_overlap = (
    train_sources
    &
    val_sources
)

if paper_overlap:
    raise RuntimeError(
        "发现paper-level train/val泄漏: "
        f"{len(paper_overlap)}篇"
    )

print(
    "\nPaper-level split检查通过"
)

print(
    "train samples:",
    len(train_dataset)
)

print(
    "val samples:",
    len(val_dataset)
)

print(
    "train papers:",
    len(train_sources)
)

print(
    "val papers:",
    len(val_sources)
)

print(
    "train/val paper overlap:",
    len(paper_overlap)
)


# =====================================================
# DataLoader
# =====================================================

train_generator = torch.Generator()

train_generator.manual_seed(
    seed
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=use_amp,
    generator=train_generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
    pin_memory=use_amp,
)


# =====================================================
# Step统计
# =====================================================

optimizer_steps_per_epoch = math.ceil(
    len(train_loader)
    /
    gradient_accumulation_steps
)

total_optimizer_steps = (
    optimizer_steps_per_epoch
    *
    epochs
)

print(
    "\ntrain microsteps / epoch:",
    len(train_loader)
)

print(
    "val microsteps / epoch:",
    len(val_loader)
)

print(
    "optimizer updates / epoch:",
    optimizer_steps_per_epoch
)

print(
    "总optimizer updates:",
    total_optimizer_steps
)


# =====================================================
# Optimizer
#
# 与普通SFT保持AdamW。
# =====================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)


# =====================================================
# AMP
# =====================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# =====================================================
# Validation
# =====================================================

@torch.no_grad()
def evaluate(
    model,
    data_loader,
):
    model.eval()

    total_weighted_loss = 0.0
    total_supervised_tokens = 0

    for batch in data_loader:

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=use_amp,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=use_amp,
        )

        supervised_tokens = int(
            (
                labels
                !=
                -100
            ).sum().item()
        )

        if supervised_tokens <= 0:
            continue

        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            _, loss = model(
                input_ids,
                labels,
            )

        loss_value = float(
            loss.detach().item()
        )

        total_weighted_loss += (
            loss_value
            *
            supervised_tokens
        )

        total_supervised_tokens += (
            supervised_tokens
        )

    if total_supervised_tokens == 0:
        raise RuntimeError(
            "Validation没有任何有效监督token"
        )

    return (
        total_weighted_loss
        /
        total_supervised_tokens
    )


# =====================================================
# 保存Checkpoint
# =====================================================

def save_checkpoint(
    save_file,
    epoch,
    train_loss,
    val_loss,
    best_val_loss,
):
    torch.save(
        {
            "model":
                model.state_dict(),

            "config":
                config.__dict__,

            "epoch":
                epoch,

            "training_stage":
                "rag_sft",

            "source_checkpoint":
                str(
                    CHECKPOINT_FILE
                ),

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "best_val_loss":
                best_val_loss,

            "learning_rate":
                learning_rate,

            "batch_size":
                batch_size,

            "gradient_accumulation_steps":
                gradient_accumulation_steps,

            "effective_batch_size":
                effective_batch_size,

            "max_grad_norm":
                max_grad_norm,

            "train_samples":
                len(
                    train_dataset
                ),

            "val_samples":
                len(
                    val_dataset
                ),

            "train_papers":
                len(
                    train_sources
                ),

            "val_papers":
                len(
                    val_sources
                ),

            "paper_split_seed":
                seed,

            "paper_train_ratio":
                0.95,

        },
        save_file,
    )


# =====================================================
# 开始训练
# =====================================================

print(
    "\n================================="
)

print(
    "开始RAG-aware SFT训练"
)

print(
    "================================="
)


best_val_loss = float(
    "inf"
)

best_epoch = None


for epoch_index in range(
    epochs
):

    epoch_number = (
        epoch_index
        +
        1
    )

    model.train()

    # 用监督Answer token做加权统计，
    # 让epoch loss更可比较。
    total_weighted_train_loss = 0.0
    total_supervised_train_tokens = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulation_count = 0

    for step, batch in enumerate(
        train_loader
    ):

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=use_amp,
        )

        labels = batch[
            "labels"
        ].to(
            device,
            non_blocking=use_amp,
        )

        supervised_tokens = int(
            (
                labels
                !=
                -100
            ).sum().item()
        )

        if supervised_tokens <= 0:
            raise RuntimeError(
                "训练batch没有监督Answer token"
            )

        # =============================================
        # Forward
        # =============================================

        with torch.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            _, loss = model(
                input_ids,
                labels,
            )

        raw_loss = float(
            loss.detach().item()
        )

        if not math.isfinite(
            raw_loss
        ):
            raise RuntimeError(
                "训练出现NaN/Inf loss: "
                f"epoch={epoch_number}, "
                f"step={step}, "
                f"loss={raw_loss}"
            )

        total_weighted_train_loss += (
            raw_loss
            *
            supervised_tokens
        )

        total_supervised_train_tokens += (
            supervised_tokens
        )

        # =============================================
        # Gradient Accumulation
        # =============================================

        scaled_loss = (
            loss
            /
            gradient_accumulation_steps
        )

        scaler.scale(
            scaled_loss
        ).backward()

        accumulation_count += 1

        is_last_batch = (
            step + 1
            ==
            len(train_loader)
        )

        should_step = (
            accumulation_count
            ==
            gradient_accumulation_steps
            or
            is_last_batch
        )

        if should_step:

            # -----------------------------------------
            # 如果最后一个accumulation window不足4个batch，
            # 前面每个loss仍除以4，会导致梯度偏小。
            #
            # 这里在optimizer step前进行比例补偿。
            # -----------------------------------------

            if (
                accumulation_count
                <
                gradient_accumulation_steps
            ):

                correction = (
                    gradient_accumulation_steps
                    /
                    accumulation_count
                )

                for parameter in model.parameters():

                    if parameter.grad is not None:
                        parameter.grad.mul_(
                            correction
                        )

            # -----------------------------------------
            # AMP下先unscale，再做gradient clipping
            # -----------------------------------------

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation_count = 0

        # =============================================
        # 日志
        # =============================================

        if (
            step % log_interval == 0
            or
            step + 1 == len(train_loader)
        ):

            running_train_loss = (
                total_weighted_train_loss
                /
                total_supervised_train_tokens
            )

            print(
                f"epoch {epoch_number}/{epochs}",
                f"step {step + 1}/{len(train_loader)}",
                f"batch_loss {raw_loss:.4f}",
                f"running_loss {running_train_loss:.4f}",
            )


    # =================================================
    # Train loss
    # =================================================

    train_loss = (
        total_weighted_train_loss
        /
        total_supervised_train_tokens
    )


    # =================================================
    # Validation
    # =================================================

    print(
        "\n开始Validation..."
    )

    val_loss = evaluate(
        model,
        val_loader,
    )


    print(
        "\n================================="
    )

    print(
        "epoch:",
        epoch_number
    )

    print(
        "RAG-SFT train loss:",
        train_loss
    )

    print(
        "RAG-SFT val loss:",
        val_loss
    )


    # =================================================
    # 保存当前epoch
    # =================================================

    epoch_file = (
        SAVE_DIR
        /
        f"yanlongllm_cnt_ragsft_epoch{epoch_number}.pt"
    )

    current_best_for_metadata = min(
        best_val_loss,
        val_loss,
    )

    save_checkpoint(
        save_file=epoch_file,
        epoch=epoch_number,
        train_loss=train_loss,
        val_loss=val_loss,
        best_val_loss=current_best_for_metadata,
    )

    print(
        "Checkpoint保存:"
    )

    print(
        epoch_file
    )


    # =================================================
    # Best checkpoint
    # =================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch_number

        save_checkpoint(
            save_file=BEST_CHECKPOINT_FILE,
            epoch=epoch_number,
            train_loss=train_loss,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
        )

        print(
            "Best checkpoint更新:"
        )

        print(
            BEST_CHECKPOINT_FILE
        )

    else:

        print(
            "Best checkpoint未更新"
        )


# =====================================================
# 完成
# =====================================================

print(
    "\n================================="
)

print(
    "RAG-aware SFT训练完成"
)

print(
    "================================="
)

print(
    "模型保存目录:"
)

print(
    SAVE_DIR
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best val loss:",
    best_val_loss
)

print(
    "Best checkpoint:"
)

print(
    BEST_CHECKPOINT_FILE
)

print(
    "\n下一步:"
)

print(
    "22_test_rag_sft.py"
)

print(
    "================================="
)
