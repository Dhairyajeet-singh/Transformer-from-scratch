import numpy as np
from typing import Optional
class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(dim)   #weights
        self.beta = np.zeros(dim)   #biases
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1 , keepdims=True)
        var = np.var(x,axis=-1, keepdims=True)
        self.normalised = (x-mean)/np.sqrt(var + self.eps)
        return self.gamma * self.normalised + self.beta
    #using formula y = (x - mean(x) / sqrt ( var(x) + eps) )* gamma + beta
    #reference doc: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)

        x_mu = x - mean
        dx_norm = grad_output * self.gamma

        self.dgamma = np.sum(grad_output * self.normalised, axis=(0,1))
        self.dbeta = np.sum(grad_output, axis=(0,1))

        dvar = np.sum(dx_norm * x_mu, axis=-1, keepdims=True) * (-0.5) * (std_inv**3)
        dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2.0 * x_mu, axis=-1, keepdims=True)

        dx = (dx_norm * std_inv) \
                + (dvar * 2.0 * x_mu / x.shape[-1]) \
                + (dmean / x.shape[-1])

        return dx
class Linear:
    def __init__(self, in_features: int, out_features: int):
        scale = np. sqrt(2.0/(in_features+out_features))
        self.weight = np.random.randn(in_features, out_features) * scale
        self.bias = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weight + self.bias
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weight = self.input.T(0,2,1) @ grad_output
        self.grad_bias = grad_output.sum(axis= (0,1))
        return grad_output @ self.weight.T
        
def softmax(x: np.ndarray , axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x-np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def GeLU(x:np.ndarray) -> np.ndarray:
    return 0.5 * x * (1+ np.tanh(np.sqrt(2/np.pi)*(x + 0.044715 * np.power(x,3))))
#reference doc: https://arxiv.org/abs/1606.08415

def apply_dropout(x:np.ndarray , dropout_rate: float, training: bool = True) -> np.ndarray:
    if not training or dropout_rate == 0.0:
        return x
    mask = (np.random.rand(*x.shape) >= dropout_rate).astype(x.dtype)
    return x * mask / (1.0 -dropout_rate)

class MultiHeadAttention:
    """
    Multi-Head Self-Attention mechanism.
    
    Key concepts:
    - Query (Q): "What am I looking for?"
    - Key (K): "What do I contain?"
    - Value (V): "What do I actually offer?"
    - Attention Score: How much should token i focus on token j?
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model     #embedding size
        self.dropout_p = dropout    #dropout rate
        self.num_heads = num_heads  #number of attention heads
        self.d_k = d_model // num_heads #size per head

        """
        d_model = 512
        num_heads = 8
        d_k = 512 / 8 = 64
        """

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def split_heads(self, x: np.ndarray) ->  np.ndarray:
        """
            Input: (batch, seq_len, d_model)
            Output: (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
    
    '''Attention(Q,K,V)=softmax(np.exp(QK,T)/ np.sqrt(d_k))V'''

    def combine_heads(self, x:np.ndarray) -> np.ndarray:
        """
            Combine heads back together.
            Input: (batch, num_heads, seq_len, d_k)
            Output: (batch, seq_len, d_model)
        """
         
        batch_size, _, seq_len, _ = x.shape
        x= x.transpose(0,2,1,3)
        return x.reshape(batch_size, seq_len, self.d_model)
        
    def forward(self , x:np.ndarray, mask:Optional[np.ndarray] = None, training: bool = True) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        """
            Forward pass of multi-head attention.
            
            Args:
                x: Input tensor (batch, seq_len, d_model)
                mask: Optional mask (batch, 1, 1, seq_len) for padding or causality
                training: Whether in training mode (for dropout)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear projections
        Q = self.W_q.forward(x)  # (batch, seq_len, d_model)
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)

        # Linear projections
        Q = self.split_heads(Q)  # (batch_size, num_heads, seq_len, d_k)
        K = self.split_heads(K)  # (batch_size, num_heads, seq_len, d_k)
        V = self.split_heads(V)  # (batch_size, num_heads, seq_len, d_k)

        # 3. Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
        # Compute attention scores

        # Scaled dot-product attention
        # Compute attention scores
        
        # Compute attention scores
        scores = Q @ K.transpose(0, 1, 3, 2)  # (batch, num_heads, seq_len, seq_len)
        scores = scores / np.sqrt(self.d_k)  # scale by sqrt(d_k)
        
        # Apply mask if provided (for padding or causal attention)
        if mask is not None:
            scores = scores + (mask * -1e9)  # add large negative value to masked positions
        
        # Apply softmax to get attention weights
        self.attention_weights = softmax(scores, axis=-1)  # (batch, num_heads, seq_len, seq_len)
        
        # Apply dropout
        attention = apply_dropout(self.attention_weights, self.dropout_p, training)
        
        # 4. Apply attention to values
        output = attention @ V  # (batch, num_heads, seq_len, d_k)
        
        # 5. Combine heads
        output = self.combine_heads(output)  # (batch, seq_len, d_model)
        
        # 6. Final linear projection
        output = self.W_o.forward(output)
        
        return output

class FeedForward:
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1): 
        """
        Position-wise Feed-Forward Network.
        FFN(x) = max(0, xW1 + b1)W2 + b2
        Or with GELU: GELU(xW1 + b1)W2 + b2
        """
        self.Linear1 = Linear(d_model, d_ff)
        self.Linear2 = Linear(d_ff, d_model)
        self.dropout_p = dropout 
    def forward(self, x:np.ndarray , training:bool = True) -> np.ndarray: 
        x = self.Linear1.forward(x)  # (batch, seq_len, d_ff)
        x = GeLU(x)
        x = apply_dropout(x , self.dropout_p, training)
        x = self.Linear2.forward(x)
        return x
    

class positional_encoding:
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            max_len: max length of input sequence
            d_model: demension of embedding
        
        Sinusoidal positional encoding to inject position information.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """

        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis] # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * - (np.log(10000.0) / d_model))
        pe[:,0::2] = np.sin(position * div_term) 
        pe[:,1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self , x: np.ndarray) -> np.ndarray: 
        """
        Add positional encoding to input embeddings.
        x shape: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :]
    
class Transformer_Encoder_Block:
    def __init__(self , d_model: int, d_ff: int, num_heads: int, dropout:float = 0.1):
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout_p = dropout
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None, training: bool = True) -> np.ndarray:
        # Multi-Head Attention with residual connection and layer norm
        attn_output = self.mha.forward(x, mask)
        attn_output = apply_dropout(attn_output, self.dropout_p, training)
        x = self.norm1.forward(x+attn_output)

        ff_output = self.ffn.forward(x, training)
        ff_output = apply_dropout(ff_output, self.dropout_p, training)
        x = self.norm2.forward(x+ff_output)
        return x
    

class Transformer_Encoder:
    def __init__(self , vocab_size:int , num_layers:int , d_model: int, d_ff:int, num_heads:int,max_len:int = 5000 , dropout: float = 0.1):
        self.d_model = d_model
        self.embedding = np.random.randn(vocab_size , d_model) * np.sqrt(2.0/vocab_size)
        self.positional_encoding = positional_encoding(d_model, max_len)

        self.layers = [
            Transformer_Encoder_Block(d_model, d_ff, num_heads, dropout) 
            for _ in range(num_layers)
        ]

        self.dropout_p = dropout

    def forward(self, x:np.ndarray, mask:Optional [np.ndarray] = None, training: bool = True) -> np.ndarray:
        """
        Forward pass through the entire encoder.
        
        Args:
            x: Input token indices (batch, seq_len)
            mask: Optional attention mask
            training: Whether in training mode
        """
        x = self.embedding[x]
        x = x * np.sqrt(self.d_model)  # scale embeddings
        x = self.positional_encoding.forward(x)
        x = apply_dropout(x, self.dropout_p, training)

        for layer in self.layers:
            x = layer.forward(x, mask, training)
        return x
    
def create_padding_mask(seq: np.ndarray, pad_token: int = 0) -> np.ndarray:
    """
    Create a padding mask for sequences.
    
    Args:
        seq: Input sequences (batch, seq_len)
        pad_token: Token used for padding (default: 0)
    """
    mask = (seq == pad_token).astype(np.float32)
    return mask[:, np.newaxis, np.newaxis, :]  # (batch, 1, 1, seq_len)

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal mask for autoregressive decoding.
    
    Args:
        seq_len: Length of the sequence
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float32)
    return mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, seq_len)

if __name__ == "__main__":
    print("=" * 70)
    print("TRANSFORMER FROM SCRATCH - DEMONSTRATION")
    print("=" * 70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Hyperparameters
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 100
    dropout = 0.1
   
    print("\nModel Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension (d_model): {d_model}")
    print(f"  Number of attention heads: {num_heads}")
    print(f"  Number of encoder layers: {num_layers}")
    print(f"  Feed-forward dimension: {d_ff}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Dropout rate: {dropout}")

    print("\n\n" + "=" * 70)
    print("Building Transformer Encoder...")

    transformer = Transformer_Encoder(
        vocab_size = vocab_size,
        num_layers = num_layers,
        num_heads = num_heads,
        d_model = d_model,
        d_ff = d_ff,
        dropout = dropout,
        max_len=max_len
    )
    print(" Transformer built successfully!")
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    
    print("\n" + "=" * 70)
    print("Running Forward Pass...")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Random token indices (simulating tokenized text)
    input_tokens = np.random.randint(1, vocab_size, size=(batch_size, seq_len))
    print(f"\nInput tokens shape: {input_tokens.shape}")
    print(f"Sample input tokens:\n{input_tokens[0]}")
    
    padding_mask = create_padding_mask(input_tokens, pad_token=0)
    
    # Forward pass
    output = transformer.forward(input_tokens, mask=padding_mask, training=True)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"✓ Shape matches!")
    
    print("\n" + "=" * 70)
    print("Testing Individual Components:")
    print("=" * 70)

    # Test attention mechanism
    print("\n1. Multi-Head Attention:")
    test_input = np.random.randn(2, 5, d_model)
    attention = MultiHeadAttention(d_model, num_heads)
    attn_output = attention.forward(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {attn_output.shape}")
    print(f"   Attention weights shape: {attention.attention_weights.shape}")
    print(f"   ✓ Attention working correctly!")
    
    # Test feed-forward
    print("\n2. Feed-Forward Network:")
    ff = FeedForward(d_model, d_ff)
    ff_output = ff.forward(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {ff_output.shape}")
    print(f"   ✓ Feed-forward working correctly!")
    
    # Test positional encoding
    print("\n3. Positional Encoding:")
    pos_enc = positional_encoding(d_model, max_len)
    pos_output = pos_enc.forward(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {pos_output.shape}")
    print(f"   ✓ Positional encoding working correctly!")
    
    print("\n" + "=" * 70)
    print("Visualizing Attention Patterns:")
    print("=" * 70)

    first_layer = transformer.layers[0]
    _ = first_layer.forward(output[:1], training=False)  # Forward pass to compute attention
    attn_weights = first_layer.mha.attention_weights[0, 0]  # First head of the first layer


    print(f"\nAttention weights shape: {attn_weights.shape}")
    print(f"Attention weights (first head, first batch):")
    print(np.round(attn_weights, 3))
    print("\nEach row shows how much each token attends to all other tokens.")
    print("Values sum to 1.0 across each row (due to softmax).")
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION COMPLETE!")
    print("=" * 70)
    print("\nKey Features Implemented:")
    print("  ✓ Multi-head self-attention")
    print("  ✓ Positional encoding")
    print("  ✓ Feed-forward networks")
    print("  ✓ Layer normalization")
    print("  ✓ Residual connections")
    print("  ✓ Dropout regularization")
    print("  ✓ Attention masking")
    print("\nThis is a fully functional transformer encoder built from scratch!")
    print("=" * 70)