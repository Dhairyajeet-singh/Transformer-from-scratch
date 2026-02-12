import numpy as np

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

        def forward(x: np.ndarray) -> np.ndarray:
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
        self.d_k = d_model // num_heads

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def split_heads(self, x: np.ndarray) ->  np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_k)
    