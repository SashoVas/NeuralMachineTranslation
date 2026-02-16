import torch


class Attention(torch.nn.Module):
    def __init__(self, d_model, d_keys, d_values, masked=False, context_length=128):
        super().__init__()
        self.q = torch.nn.Linear(d_model, d_keys, bias=False)
        self.k = torch.nn.Linear(d_model, d_keys, bias=False)
        self.v = torch.nn.Linear(d_model, d_values, bias=False)
        self.o = torch.nn.Linear(d_values, d_model, bias=False)
        self.scale = 1/(d_keys ** 0.5)
        self.masked = masked
        if masked:
            self.register_buffer("mask", torch.tril(
                torch.ones(context_length, context_length)) == 0)

    def forward(self, input):
        batch_size, seq_len, embed_dim = input.shape
        q_proj = self.q(input)
        k_proj = self.k(input)
        v_proj = self.v(input)

        logits = (q_proj @ k_proj.transpose(1, 2))*self.scale
        if self.masked:
            logits = logits.masked_fill(
                self.mask[:seq_len, :seq_len], -float('inf'))

        probs = torch.nn.functional.softmax(logits, dim=2)
        res = probs@v_proj

        return self.o(res)


class MultiHeadAttn(torch.nn.Module):
    def __init__(self, d_model, d_keys, d_values, n_heads=4, masked=False, context_length=128):
        super().__init__()
        self.q = torch.nn.Linear(d_model, d_keys*n_heads, bias=False)
        self.k = torch.nn.Linear(d_model, d_keys*n_heads, bias=False)
        self.v = torch.nn.Linear(d_model, d_values*n_heads, bias=False)
        self.o = torch.nn.Linear(d_values*n_heads, d_model, bias=False)
        self.scale = 1/(d_keys ** 0.5)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_keys = d_keys
        self.d_values = d_values
        self.masked = masked
        if masked:
            self.register_buffer("mask", torch.tril(
                torch.ones(context_length, context_length)) == 0)

    def forward(self, input):
        batch_size, seq_length, embed_dim = input.shape
        q_proj = self.q(input).view(batch_size, seq_length,
                                    self.n_heads, self.d_keys).transpose(1, 2)
        k_proj = self.k(input).view(batch_size, seq_length,
                                    self.n_heads, self.d_keys).permute(0, 2, 3, 1)
        v_proj = self.v(input).view(batch_size, seq_length,
                                    self.n_heads, self.d_values).transpose(1, 2)

        logits = (q_proj @ k_proj)*self.scale
        if self.masked:
            logits = logits.masked_fill(
                self.mask[:seq_length, :seq_length], -float('inf'))

        probs = torch.nn.functional.softmax(logits, dim=3)
        res_heads = probs@v_proj
        res = self.o(res_heads.transpose(1, 2).flatten(2, 3))

        return res


class OptimizedMultiHeadAttn(torch.nn.Module):
    def __init__(self, d_model, d_keys, n_heads, context_length):
        super().__init__()
        self.d_keys = d_keys
        self.n_heads = n_heads
        self.d_model = d_model
        self.scale = 1/(d_keys ** 0.5)

        self.attn = torch.nn.Linear(d_model, 3*d_keys*n_heads)
        self.proj = torch.nn.Linear(n_heads*d_keys, d_model)

        self.register_buffer("mask", torch.tril(
            torch.ones(context_length, context_length)) == 0)

    def forward(self, input):
        batch_size, seq_len, embed_dim = input.shape
        qkv = self.attn(input)
        q_proj, k_proj, v_proj = torch.split(
            qkv, self.d_keys*self.n_heads, dim=2)
        q_proj = q_proj.view(batch_size, seq_len,
                             self.n_heads, self.d_keys).transpose(1, 2)
        k_proj = k_proj.view(batch_size, seq_len,
                             self.n_heads, self.d_keys).transpose(1, 2)
        v_proj = v_proj.view(batch_size, seq_len,
                             self.n_heads, self.d_keys).transpose(1, 2)

        # logits = (q_proj @ k_proj.transpose(-2, -1))*self.scale
        # logits = logits.masked_fill(self.mask[:seq_len,:seq_len],-float('inf'))
        # probs = torch.nn.functional.softmax(logits,dim=3)
        # res_heads = probs@v_proj

        res_heads = torch.nn.functional.scaled_dot_product_attention(
            q_proj, k_proj, v_proj, is_causal=True)

        res = self.proj(res_heads.transpose(1, 2).flatten(2, 3))

        return res


class Transformer(torch.nn.Module):
    def __init__(self, d_in, d_keys, d_values, dropout_prob=0.2, context_length=128, n_heads=None):
        super().__init__()
        if n_heads == None:
            self.attn = Attention(d_in, d_keys, d_values,
                                  True, context_length=context_length)
        else:
            # MultiHeadAttn(d_in,d_keys,d_values,n_heads=n_heads,masked=True,context_length=context_length)
            self.attn = OptimizedMultiHeadAttn(
                d_in, d_keys, n_heads, context_length)
        self.ln1 = torch.nn.LayerNorm(d_in)
        self.dropout1 = torch.nn.Dropout(dropout_prob)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_in, 4*d_in),
            # Used in gpt2 instead of relu
            torch.nn.GELU(),
            torch.nn.Linear(d_in*4, d_in)
        )
        self.dropout2 = torch.nn.Dropout(dropout_prob)

        self.ln2 = torch.nn.LayerNorm(d_in)

    def forward(self, input):
        attn_res = self.attn(self.ln1(input))
        res1 = self.dropout1(attn_res) + input
        feed_forward_res = self.feed_forward(self.ln2(res1))
        res2 = self.dropout2(feed_forward_res) + res1
        return res2


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(0).unsqueeze(2)
        div_term = (10000.0 ** (torch.arange(0, d_model, 2)/d_model)
                    ).unsqueeze(0).unsqueeze(0)
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return x


class LanguageModel(torch.nn.Module):
    def __init__(self, num_token, d_model, d_keys, d_values, word2ind, endToken, dropout_prob=0.2, context_length=128, n_heads=None, transformer_layers=1):
        super().__init__()
        self.word2ind = word2ind
        self.endToken = endToken
        self.pos_embed = PositionalEncoding(d_model, max_len=context_length)
        self.embed = torch.nn.Embedding(num_token, d_model)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.context_len = context_length
        self.transformer = torch.nn.Sequential(*[
            Transformer(d_model, d_keys, d_values, dropout_prob=dropout_prob, context_length=context_length, n_heads=n_heads) for i in range(transformer_layers)
        ])
        self.proj_layer = torch.nn.Linear(d_model, num_token)

        # Weight sharing
        self.embed.weight = self.proj_layer.weight

    def forward(self, input):
        embed_res = self.dropout(self.pos_embed(self.embed(input)))
        transformer_res = self.transformer(embed_res)
        logits = self.proj_layer(transformer_res)

        return logits

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(torch.load(fileName))

    def generate(self, input, device="cuda:0", mode="softmax", mask=None):
        output = []
        self.eval()
        with torch.no_grad():
            for i in range(self.context_len - len(input)):
                start = torch.tensor([input]).to(device=device)
                logits = self.forward(start)
                next_symbol_vector = logits[0][-1]
                if mask is not None:
                    next_symbol_vector = next_symbol_vector.masked_fill(
                        mask, -100)
                if mode == "softmax":
                    next = torch.multinomial(torch.nn.functional.softmax(
                        next_symbol_vector, dim=-1), 1)[0].item()
                elif mode == "argmax":
                    next = torch.argmax(next_symbol_vector).item()
                output.append(next)
                input.append(next)
                if next == self.word2ind[self.endToken]:
                    break
        return output
