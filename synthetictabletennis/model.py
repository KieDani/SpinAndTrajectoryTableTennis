import torch
import torch.nn as nn
import einops as eo
import math

from helper import table_points, table_lines, HEIGHT, WIDTH


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.0, bias = True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, proj_bias = True, attn_drop = 0.0, proj_drop = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight, gain=1)
        nn.init.xavier_uniform_(self.proj.weight, gain=1.414)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, mask=None):
        '''Forward pass through the transformer
        Args:
            x (torch.Tensor): Tensor of shape (batch_size*seq_len, num_pos, dim)
            mask (torch.Tensor): Tensor of shape (batch_size*seq_len, num_pos) indicating at which positions there is padding and which not
        '''
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            attn = attn + mask[:, None, None, :] + mask[:, None, :, None]  # Add mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def staticPositionalEncoding(d_model, length):
    """ Create positional encoding.
    Args:
        d_model: int, dimension of the model
        length: int, length of positions
    """
    if d_model % 2 != 0:
        raise ValueError("Even dimension is required")
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        """
        Initialize the RotaryPositionalEmbedding class.

        Parameters:
        dim (int): Dimension of the input embeddings.
        """
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = dim
        # Precompute sinusoidal frequencies
        self.inv_freq = nn.Parameter(1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)), requires_grad=False)

    def forward(self, x):
        """
        Apply rotary positional embeddings to the input tensor.

        Parameters:
        x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, dim).
        seq_len (int): Length of the sequence.

        Returns:
        torch.Tensor: Tensor with rotary positional embeddings applied.
        """
        B, num_heads, T, D = x.shape
        assert D == self.dim, "Input dimension does not match embedding dimension."

        # Calculate rotation angles using precomputed inverse frequencies
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.einsum('i,j->ij', pos, self.inv_freq)  # (T, D/2)
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)

        # Split x into uneven and even parts
        x_uneven = x[..., 0::2]
        x_even = x[..., 1::2]

        # Apply rotary embedding transformation
        x_rotated_uneven = x_uneven * cos - x_even * sin
        x_rotated_even = x_uneven * sin + x_even * cos

        # Interleave the rotated components back together
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_rotated_uneven
        x_rotated[..., 1::2] = x_rotated_even

        return x_rotated


class TableEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TableEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = 2 # x, y in image space
        self.num_tokens = 13 # number of predefined table positions
        self.fc1 = nn.Linear(self.in_dim, self.dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.dim, self.dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        '''Transforms the table tokens into the correct shape'''
        B, N, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        assert N <= self.num_tokens, "Number of tokens exceeds the number of predefined table positions."
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class BallEmbedding(nn.Module):
    def __init__(self, embed_dim, in_dim=2):
        super(BallEmbedding, self).__init__()
        self.dim = embed_dim
        self.in_dim = in_dim # usually x, y in image space
        self.fc1 = nn.Linear(self.in_dim, self.dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.dim, self.dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        '''Transforms the ball position into the correct shape'''
        B, N, D = x.shape
        assert D == self.in_dim, "Input dimension does not match embedding dimension."
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class AttentionWithRotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rotary_emb = RotaryPositionalEmbedding(dim // num_heads)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight, gain=1)
        nn.init.xavier_uniform_(self.proj.weight, gain=1)
        if self.qkv.bias is not None:
            nn.init.constant_(self.qkv.bias, 0)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, mask, num_cls_token=0):
        '''Forward pass through the transformer with applying rotary positional embeddings. Use this for the ball positions
        Args:
            x (torch.Tensor): Tensor of shape (batch_size*num_pos, num_heads, seq_len, dim)
            mask (torch.Tensor): Tensor of shape (batch_size*num_pos, seq_len) indicating at which times there is padding and which not
            num_cls_token (int): Number of cls tokens in the input tensor. Don't apply rotary positional embeddings to these tokens.
        '''
        B, N, C = x.shape

        # Generate q, k, v projections
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: (B, num_heads, N, head_dim)

        # Apply rotary positional embeddings to q and k
        if num_cls_token > 0:
            c_q, q = q[:, :, :num_cls_token], q[:, :, num_cls_token:]
            c_k, k = k[:, :, :num_cls_token], k[:, :, num_cls_token:]
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        if num_cls_token > 0:
            q = torch.cat((c_q, q), dim=2)
            k = torch.cat((c_k, k), dim=2)

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1) # Shape: (B, num_heads, N, N)
        attn = attn + mask[:, None, None, :] + mask[:, None, :, None]  # Add mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MyHead(nn.Module):
    '''Head for the transformer to predict the rotation of the ball.'''
    def __init__(self, dim):
        super(MyHead, self).__init__()
        self.fc1 = nn.Linear(dim, dim//2)
        self.fc2 = nn.Linear(dim//2, dim//4)
        self.fc3 = nn.Linear(dim//4, 3) # 3 for the rotation axis
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight, gain=1)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        '''Forward pass through the head
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, dim) or (batch_size, seq_len, dim)
        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class SimpleStaticLayer(nn.Module):
    '''First an attention between the positions at a specific time is computed, afterwards an attention between the different times.'''
    def __init__(self, dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=True):
        super(SimpleStaticLayer, self).__init__()
        self.rotary_attention = rotary_attention
        if rotary_attention:
            self.attn = AttentionWithRotaryPositionalEmbedding(dim, num_heads, qkv_bias, attn_drop_rate)
        else:
            self.attn = Attention(dim, num_heads, qkv_bias, attn_drop_rate)
        self.mlp1 = Mlp(in_features=dim, hidden_features=dim, act_layer=nn.ReLU, drop=0.0)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self._initialize_weights()

    def _initialize_weights(self):
        pass

    def forward(self, x, mask, num_cls_token=None):
        '''Forward pass through the layer
        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, dim)
            mask (torch.Tensor): Tensor of shape (batch_size, seq_len) indicating at which times there is padding and which not
            num_cls_token (int): Number of cls tokens in the input tensor. Only used for rotary attention
        '''
        B, T, D = x.shape

        x_res = x  # Store residual
        x = self.norm1(x)  # Apply layer norm
        if self.rotary_attention:
            if num_cls_token is not None:
                x = self.attn(x, mask, num_cls_token)
            else:
                x = self.attn(x, mask, 0)
        else:
            x = self.attn(x, mask)
        x = x + x_res
        x_res = x  # Store residual
        x = self.norm2(x)
        x = self.mlp1(x)
        x = x + x_res

        return x


class FirstStage(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, mode, pos_embedding):
        super(FirstStage, self).__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = int(mlp_ratio * dim)
        self.pos_embedding = pos_embedding
        assert pos_embedding in ['rotary', 'added'], 'pos_embedding should be either "rotary" or "added"'

        # free = no table tokens, static = table tokens for static camera per sequence, dynamic = table tokens for dynamic camera per time step
        assert mode in ['free', 'static', 'dynamic', 'stacked'], 'mode should be either "free", "static", "dynamic", "stacked"'
        self.mode = mode

        if mode == 'stacked':
            self.ball_embed = BallEmbedding(dim, len(table_points) * 2 + 2)  # num table tokens * 2 + 2 for the ball token
        else:
            self.ball_embed = BallEmbedding(dim, 2)
        if mode in ['static', 'dynamic']:
            self.table_embed = TableEmbedding(dim)

        if mode in ['static']:
            self.pos_layers = nn.ModuleList([
                SimpleStaticLayer(dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=True)
                for _ in range(4)
            ])
        elif mode in ['dynamic']:
            self.pos_layers = nn.ModuleList([
                SimpleStaticLayer(dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=False)
                for _ in range(4)
            ])

        rotary_attention = True if pos_embedding == 'rotary' else False
        self.layers = nn.ModuleList([
            SimpleStaticLayer(dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=rotary_attention)
            for _ in range(self.depth)
        ])

        self.position_head = MyHead(dim)

    def forward(self, ball_pos, table_pos, mask):
        '''Forward pass through the transformer. Masks are already expected to be in the correct format.
        Args:
            ball_pos (torch.Tensor): Tensor of shape (batch_size, seq_len, 2)
            table_pos (torch.Tensor): Tensor of shape (batch_size, num_positions, 2)
            mask (torch.tensor): Tensor of shape (batch_size, seq_len) indicating at which times there is padding and which not
        '''
        B, T, _ = ball_pos.shape
        D = self.dim

        if self.mode == 'stacked':
            table_pos = eo.rearrange(table_pos, 'b n d -> b (n d)')
            table_pos = eo.repeat(table_pos, 'b n -> b t n', t=T)
            ball_pos = torch.cat((ball_pos, table_pos), dim=2)

        # Embedding
        ball_pos = self.ball_embed(ball_pos)
        if self.mode in ['static', 'dynamic']:
            table_pos = self.table_embed(table_pos)
        x = ball_pos

        if self.mode == 'static':
            # Add table positions (fixed camera version)
            _, N, _ = table_pos.shape
            x = torch.cat((table_pos, x), dim=1)
            # Adjust mask for table positions
            mask_tmp = torch.zeros((B, N + T), device=ball_pos.device)
            mask_tmp[:, N:] = mask
            mask = mask_tmp
            for layer in self.pos_layers:
                x = layer(x, mask)
            x = x[:, N:, :]  # Only take the ball position tokens -> (B, T, D)
            mask = mask[:, N:]  # Only keep the mask for the ball position tokens -> (B, T)
        elif self.mode == 'dynamic':
            # Add table position (dynamic camera version)
            _, N, _ = table_pos.shape
            table_pos = table_pos.unsqueeze(1).expand(B, T, N, D)
            x = x.unsqueeze(2)
            x = torch.cat((x, table_pos), dim=2)
            x = eo.rearrange(x, 'b t n d -> (b t) n d')
            for layer in self.pos_layers:
                x = layer(x, None)
            x = eo.rearrange(x, '(b t) n d -> b t n d', b=B)
            x = x[:, :, 0, :]  # Only take the ball position tokens -> (B, T, D)

        if self.pos_embedding != 'rotary':  # Add positional encoding
            x = x + staticPositionalEncoding(D, T).to(ball_pos.device)

        for layer in self.layers:
            x = layer(x, mask)

        positions = self.position_head(x)
        return positions, x


class SingleStageModel(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, mode, pos_embedding):
        super(SingleStageModel, self).__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = int(mlp_ratio * dim)
        self.pos_embedding = pos_embedding
        assert pos_embedding in ['rotary', 'added'], 'pos_embedding should be either "rotary" or "added"'

        # free = no table tokens, static = table tokens for static camera per sequence, dynamic = table tokens for dynamic camera per time step
        assert mode in ['free', 'static', 'dynamic', 'stacked'], 'mode should be either "free", "static", "dynamic", "stacked"'
        self.mode = mode

        if mode == 'stacked':
            self.ball_embed = BallEmbedding(dim, len(table_points)*2+2) # num table tokens * 2 + 2 for the ball token
        else:
            self.ball_embed = BallEmbedding(dim, 2)
        if mode in ['static', 'dynamic']:
            self.table_embed = TableEmbedding(dim)

        if mode in ['static']:
            self.pos_layers = nn.ModuleList([
                SimpleStaticLayer(dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=True)
                for _ in range(4)
            ])
        elif mode in ['dynamic']:
            self.pos_layers = nn.ModuleList([
                SimpleStaticLayer(dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=False)
                for _ in range(4)
            ])

        rotary_attention = True if pos_embedding == 'rotary' else False
        self.layers = nn.ModuleList([
            SimpleStaticLayer(dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=rotary_attention)
            for _ in range(depth)
        ])

        self.cls_token = nn.Parameter(torch.empty(1, 1, dim), requires_grad=True)
        nn.init.xavier_uniform_(self.cls_token)

        self.rotation_head = MyHead(dim)
        self.position_head = MyHead(dim)

    def forward(self, ball_pos, table_pos, mask):
        '''Forward pass through the transformer
        Args:
            ball_pos (torch.Tensor): Tensor of shape (batch_size, seq_len, 2)
            table_pos (torch.Tensor): Tensor of shape (batch_size, num_positions, 2)
            mask (torch.tensor): Tensor of shape (batch_size, seq_len) indicating at which times there is padding and which not
        '''
        B, T, _ = ball_pos.shape
        D = self.dim

        # Transform the mask such that it can be added before the softmax operation.
        if mask.min() == 0 and mask.max() == 1:
            mask = torch.where(mask == 0, torch.tensor(-1e-9, device=mask.device), torch.tensor(0.0, device=mask.device))
        elif mask.max() == 0 and mask.min() < -1e-8:
            mask = mask
        else:
            raise ValueError('wrong format for masks. Should be 0, 1 or -1e-9, 0.')

        if self.mode == 'stacked':
            table_pos = eo.rearrange(table_pos, 'b n d -> b (n d)')
            table_pos = eo.repeat(table_pos, 'b n -> b t n', t=T)
            ball_pos = torch.cat((ball_pos, table_pos), dim=2)

        # Embedding
        ball_pos = self.ball_embed(ball_pos)
        if self.mode in ['static', 'dynamic']:
            table_pos = self.table_embed(table_pos)
        x = ball_pos

        if self.mode == 'static':
            # Add table positions (fixed camera version)
            _, N, _ = table_pos.shape
            x = torch.cat((table_pos, x), dim=1)
            # Adjust mask for table positions
            mask_tmp = torch.zeros((B, N + T), device=ball_pos.device)
            mask_tmp[:, N:] = mask
            mask = mask_tmp
            for layer in self.pos_layers:
                x = layer(x, mask)
            x = x[:, N:, :]  # Only take the ball position tokens -> (B, T, D)
            mask = mask[:, N:]  # Only keep the mask for the ball position tokens -> (B, T)
        elif self.mode == 'dynamic':
            # Add table position (dynamic camera version)
            _, N, _ = table_pos.shape
            table_pos = table_pos.unsqueeze(1).expand(B, T, N, D)
            x = x.unsqueeze(2)
            x = torch.cat((x, table_pos), dim=2)
            x = eo.rearrange(x, 'b t n d -> (b t) n d')
            for layer in self.pos_layers:
                x = layer(x, None)
            x = eo.rearrange(x, '(b t) n d -> b t n d', b=B)
            x = x[:, :, 0, :]  # Only take the ball position tokens -> (B, T, D)

        if self.pos_embedding != 'rotary':  # Add positional encoding
            x = x + staticPositionalEncoding(D, T).to(ball_pos.device)

        # Add cls token (position 0)
        x = torch.cat((self.cls_token.expand(B, 1, D), x), dim=1)
        # Adjust mask for cls token
        mask_tmp = torch.zeros((B, T + 1), device=ball_pos.device)
        mask_tmp[:, 1:] = mask
        mask = mask_tmp

        for layer in self.layers:
            if self.pos_embedding == 'rotary':
                x = layer(x, mask, num_cls_token=1)
            else:
                x = layer(x, mask)

        rot = x[:, 0, :]  # Only take the cls token -> (B, D)
        rot = self.rotation_head(rot)
        pos = x[:, 1:, :]
        pos = self.position_head(pos)

        return rot, pos


class MultiStageModel(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, mode, pos_embedding, use_skipconnection=False):
        super(MultiStageModel, self).__init__()
        self.dim = dim
        self.depth_secondstage = 4
        self.depth_firststage = depth - 4
        self.num_heads = num_heads
        self.mlp_dim = int(mlp_ratio * dim)
        self.mode = mode
        self.pos_embedding = pos_embedding
        assert pos_embedding in ['rotary', 'added'], 'pos_embedding should be either "rotary" or "added"'

        self.embed = BallEmbedding(self.dim, 3)
        self.firststage = FirstStage(self.dim, self.depth_firststage, num_heads, mlp_ratio, qkv_bias, attn_drop_rate, mode, pos_embedding)
        rotary_attention = True if pos_embedding == 'rotary' else False
        self.secondstage = nn.ModuleList([
            SimpleStaticLayer(self.dim, num_heads, qkv_bias, attn_drop_rate, rotary_attention=rotary_attention) for _ in range(self.depth_secondstage)
        ])

        self.cls_token = nn.Parameter(torch.empty(1, 1, dim), requires_grad=True)
        nn.init.xavier_uniform_(self.cls_token)

        self.rotation_head = MyHead(self.dim)

        # parameter that decides if the gradient for the rotation computation is backpropagated into the first stage
        self.full_backprop = False
        # if true, the second stage gets the high dimensional tokens as input insead of the 3D positions
        self.use_skipconnection = use_skipconnection

    def forward(self, ball_pos, table_pos, mask):
        '''Forward pass through the transformer
        Args:
            ball_pos (torch.Tensor): Tensor of shape (batch_size, seq_len, 2)
            table_pos (torch.Tensor): Tensor of shape (batch_size, num_positions, 2)
            mask (torch.tensor): Tensor of shape (batch_size, seq_len) indicating at which times there is padding and which not
        '''
        B, T, _ = ball_pos.shape
        D = self.dim

        # Transform the mask such that it can be added before the softmax operation.
        if mask.min() == 0 and mask.max() == 1:
            mask = torch.where(mask == 0, torch.tensor(-1e-9, device=mask.device), torch.tensor(0.0, device=mask.device))
        elif mask.max() == 0 and mask.min() < -1e-8:
            mask = mask
        else:
            raise ValueError('wrong format for masks. Should be 0, 1 or -1e-9, 0.')

        # first stage
        pos, pos_token = self.firststage(ball_pos, table_pos, mask)

        x = pos_token if self.use_skipconnection else pos

        # stop backpropagation -> rotation computation should not influence position computations
        if not self.full_backprop:
            x = x.detach()

        if not self.use_skipconnection: x = self.embed(x)

        if self.pos_embedding != 'rotary':  # Add positional encoding
            x = x + staticPositionalEncoding(D, T).to(ball_pos.device)

        # Add cls token (position 0)
        x = torch.cat((self.cls_token.expand(B, 1, D), x), dim=1)
        # Adjust mask for cls token
        mask_tmp = torch.zeros((B, T + 1), device=ball_pos.device)
        mask_tmp[:, 1:] = mask
        mask = mask_tmp

        for layer in self.secondstage:
            if self.pos_embedding == 'rotary':
                x = layer(x, mask, num_cls_token=1)
            else:
                x = layer(x, mask)

        rot = x[:, 0, :]  # Only take the cls token -> (B, D)
        rot = self.rotation_head(rot)
        return rot, pos


def get_model(name='singlestage', size='small', mode='static', pos_embedding='rotary'):
    drop_stuff = 0.0
    if name == 'singlestage':
        if size == 'small':
            return SingleStageModel(32, 8, 4, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding)
        elif size == 'base':
            return SingleStageModel(64, 12, 4, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding)
        elif size == 'large':
            return SingleStageModel(128, 16, 4, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding)
        elif size == 'huge':
            return SingleStageModel(192, 16, 8, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding)
        else:
            raise ValueError(f'Unknown model size {size}')
    elif name in ['multistage', 'connectstage']:
        use_skipconnection = True if name == 'connectstage' else False
        if size == 'small':
            return MultiStageModel(32, 8, 4, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding, use_skipconnection=use_skipconnection)
        elif size == 'base':
            return MultiStageModel(64, 12, 4, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding, use_skipconnection=use_skipconnection)
        elif size == 'large':
            return MultiStageModel(128, 16, 4, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding, use_skipconnection=use_skipconnection)
        elif size == 'huge':
            return MultiStageModel(192, 16, 8, 4, True, drop_stuff, mode=mode, pos_embedding=pos_embedding, use_skipconnection=use_skipconnection)
        else:
            raise ValueError(f'Unknown model size {size}')
    else:
        raise ValueError(f'Unknown model name {name}')



if __name__ == '__main__':
    for size in ['small', 'base', 'large', 'huge']:
        for modelname in ['singlestage', 'multistage', 'connectstage']:
            model = get_model(modelname, size, mode='stacked', pos_embedding='rotary')

            print('size:', size, 'model:', modelname)

            # Calculate total number of parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total number of parameters: {total_params}")

            # calculate only trainable parameters.
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable number of parameters: {trainable_params}")

            print('---')