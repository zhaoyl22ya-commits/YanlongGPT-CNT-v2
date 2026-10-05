import os
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# =========================
# 导入项目模型
# =========================

import importlib.util


MODEL_CODE_FILE = (
    Path(__file__).resolve().parent.parent
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
# 路径
# =========================

# 当前文件：
# 项目根目录/scripts/train/train_yanlongllm.py
# parents[2] = 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]


TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "train"
    / "train.bin"
)


VAL_FILE = (
    BASE_DIR
    / "data"
    / "train"
    / "val.bin"
)


SAVE_DIR = (
    BASE_DIR
    / "checkpoint"
)


SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# 超参数
# =========================

# RTX 4060 8GB：
# 1024 context 下使用较小物理 batch
batch_size = 2

# 梯度累积：
# effective batch = 2 × 4 = 8
gradient_accumulation_steps = 4

# 与 YanlongGPT-CNT v2 一致
block_size = 1024

# 保留原预训练学习率
learning_rate = 3e-4

# 先保持5个epoch
epochs = 5


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("device:", device)

print(
    "physical batch size:",
    batch_size
)

print(
    "gradient accumulation:",
    gradient_accumulation_steps
)

print(
    "effective batch size:",
    batch_size * gradient_accumulation_steps
)


# =========================
# Dataset
# =========================

class TokenDataset(Dataset):

    def __init__(
        self,
        file,
        block_size
    ):

        self.data = np.memmap(
            file,
            dtype=np.uint16,
            mode="r"
        )

        self.block_size = block_size


    def __len__(self):

        # 非重叠切块
        return (
            len(self.data)
            -
            self.block_size
        ) // self.block_size


    def __getitem__(self, idx):

        # 每次移动 block_size 个 token
        start = idx * self.block_size


        x = torch.tensor(
            self.data[
                start:
                start + self.block_size
            ],
            dtype=torch.long
        )


        y = torch.tensor(
            self.data[
                start + 1:
                start + self.block_size + 1
            ],
            dtype=torch.long
        )


        return x, y



# =========================
# Dataset / DataLoader
# =========================

train_dataset = TokenDataset(
    TRAIN_FILE,
    block_size
)


val_dataset = TokenDataset(
    VAL_FILE,
    block_size
)


print(
    "训练样本:",
    len(train_dataset)
)

print(
    "验证样本:",
    len(val_dataset)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size
)


# =========================
# 模型
# =========================

config = YanlongConfig(
    vocab_size=10000,
    block_size=1024,
    n_layer=8,
    n_head=8,
    n_embd=384,
    dropout=0.1
)


model = YanlongLLM(
    config
)


model.to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


print(
    "参数量:",
    sum(
        p.numel()
        for p in model.parameters()
    ) / 1e6,
    "M"
)


# =========================
# AMP 混合精度
# =========================

use_amp = (
    device == "cuda"
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


print(
    "AMP:",
    use_amp
)


# =========================
# 验证
# =========================

@torch.no_grad()
def evaluate():

    model.eval()

    losses = []


    for x, y in val_loader:

        x = x.to(device)
        y = y.to(device)


        with torch.amp.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_amp
        ):

            _, loss = model(
                x,
                y
            )


        losses.append(
            loss.item()
        )


    model.train()


    return sum(losses) / len(losses)


# =========================
# Training
# =========================

for epoch in range(epochs):


    model.train()


    total_loss = 0


    # 梯度累积需要在进入循环前清零
    optimizer.zero_grad(
        set_to_none=True
    )


    for step, (x, y) in enumerate(train_loader):


        x = x.to(device)
        y = y.to(device)


        # =====================
        # Forward
        # =====================

        with torch.amp.autocast(
            device_type=device,
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits, loss = model(
                x,
                y
            )


        # 保存原始 loss 用于打印
        raw_loss = loss.item()


        # 梯度累积：
        # 每次只反传 1/4 loss
        loss = (
            loss
            /
            gradient_accumulation_steps
        )


        scaler.scale(
            loss
        ).backward()


        # =====================
        # Optimizer step
        # =====================

        if (
            (step + 1)
            % gradient_accumulation_steps
            == 0
            or
            (step + 1)
            == len(train_loader)
        ):

            scaler.step(
                optimizer
            )

            scaler.update()


            optimizer.zero_grad(
                set_to_none=True
            )


        total_loss += raw_loss


        # =====================
        # 日志
        # =====================

        if step % 100 == 0:

            print(
                f"epoch {epoch + 1}",
                f"step {step}/{len(train_loader)}",
                f"loss {raw_loss:.4f}"
            )


        # =====================
        # 每1000步保存latest
        # =====================

        if (
            step % 1000 == 0
            and
            step != 0
        ):

            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "epoch": epoch,
                    "step": step
                },
                SAVE_DIR
                /
                "yanlongllm_cnt_latest.pt"
            )


    # =========================
    # Validation
    # =========================

    val_loss = evaluate()


    print("====================")


    print(
        "epoch:",
        epoch + 1
    )


    print(
        "train loss:",
        total_loss / len(train_loader)
    )


    print(
        "val loss:",
        val_loss
    )


    # =========================
    # 每个epoch保存checkpoint
    # =========================

    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__
        },
        SAVE_DIR
        /
        f"yanlongllm_cnt_epoch{epoch + 1}.pt"
    )


print("训练完成")