import torch
import torch.nn as nn
from torch.nn import functional as F


# =========================
# YanlongLLM 配置
# =========================

class YanlongConfig:

    def __init__(
        self,
        vocab_size=10000,
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=384,
        dropout=0.1
    ):

        self.vocab_size = vocab_size
        self.block_size = block_size

        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.dropout = dropout



# =========================
# Causal Self Attention
# =========================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        assert config.n_embd % config.n_head == 0


        self.qkv = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )


        self.proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )


        self.n_head = config.n_head

        self.dropout = nn.Dropout(
            config.dropout
        )


        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size
                )
            )
            .view(
                1,
                1,
                config.block_size,
                config.block_size
            )
        )



    def forward(self, x):

        B, T, C = x.size()


        qkv = self.qkv(x)


        q, k, v = qkv.chunk(
            3,
            dim=-1
        )


        q = q.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)


        k = k.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)


        v = v.view(
            B,
            T,
            self.n_head,
            C // self.n_head
        ).transpose(1, 2)



        att = (
            q @ k.transpose(-2, -1)
        ) / (
            k.size(-1) ** 0.5
        )


        att = att.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )


        att = F.softmax(
            att,
            dim=-1
        )


        att = self.dropout(att)


        y = att @ v


        y = y.transpose(
            1, 2
        ).contiguous().view(
            B, T, C
        )


        y = self.proj(y)


        return y



# =========================
# Feed Forward
# =========================

class MLP(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                config.n_embd,
                4 * config.n_embd
            ),

            nn.GELU(),

            nn.Linear(
                4 * config.n_embd,
                config.n_embd
            ),

            nn.Dropout(
                config.dropout
            )
        )


    def forward(self, x):

        return self.net(x)



# =========================
# Transformer Block
# =========================

class Block(nn.Module):

    def __init__(self, config):

        super().__init__()


        self.ln1 = nn.LayerNorm(
            config.n_embd
        )


        self.attn = CausalSelfAttention(
            config
        )


        self.ln2 = nn.LayerNorm(
            config.n_embd
        )


        self.mlp = MLP(
            config
        )


    def forward(self, x):

        x = x + self.attn(
            self.ln1(x)
        )


        x = x + self.mlp(
            self.ln2(x)
        )


        return x



# =========================
# YanlongLLM 主体
# =========================

class YanlongLLM(nn.Module):

    def __init__(self, config):

        super().__init__()


        self.config = config


        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )


        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )


        self.blocks = nn.ModuleList(
            [
                Block(config)
                for _ in range(config.n_layer)
            ]
        )


        self.ln_f = nn.LayerNorm(
            config.n_embd
        )


        self.head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )



    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape


        # 防止输入超过最大上下文长度
        if T > self.config.block_size:

            raise ValueError(
                f"输入长度 {T} 超过 block_size "
                f"{self.config.block_size}"
            )


        pos = torch.arange(
            0,
            T,
            device=idx.device
        )


        tok_emb = self.token_embedding(
            idx
        )


        pos_emb = self.position_embedding(
            pos
        )


        x = tok_emb + pos_emb



        for block in self.blocks:

            x = block(x)



        x = self.ln_f(x)


        logits = self.head(x)



        loss = None


        if targets is not None:


            loss = F.cross_entropy(
                logits.view(
                    -1,
                    logits.size(-1)
                ),
                targets.view(-1)
            )


        return logits, loss



# =========================
# 测试
# =========================

if __name__ == "__main__":


    config = YanlongConfig()


    model = YanlongLLM(
        config
    )


    params = sum(
        p.numel()
        for p in model.parameters()
    )


    print("========================")

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


    print(
        "YanlongLLM参数量:",
        params / 1e6,
        "M"
    )

    print("========================")


    # 使用 batch=1 测试完整 1024 context
    x = torch.randint(
        0,
        config.vocab_size,
        (1, config.block_size)
    )


    y, _ = model(x)


    print(
        "输入尺寸:",
        x.shape
    )


    print(
        "输出尺寸:",
        y.shape
    )