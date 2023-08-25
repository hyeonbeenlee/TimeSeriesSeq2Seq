# TimeSeriesSeq2Seq
Seq2Seq, Seq2Point modeling using CNN-1D, LSTM, and Attention mechanisms.  
The repo implements the following:  
- Stacked residual 1D convolution layers
- (Bidirectional) stacked LSTM layers with:  
  learnable initial states $(h_0, c_0)$, layer normalization, and the first-layer-dropout ($\xleftrightarrow{}$```torch.nn.LSTM```)
- Bahdanau and Scaled Dot Product attention in LSTM Encoder-Decoder network

Difference between ```Seq2Seq``` and ```Seq2Point``` is the decoder part.  
The former uses autoregressive LSTM decoder to generate sequence of vectors, while the latter uses MLP decoder to generate a single vector.  
See [Tutorial.ipynb](https://github.com/hyeonbeenlee/NeuralSeq2Seq/blob/main/Tutorial.ipynb) for details.

# Imports
```
import torch
import torch.nn as nn
from architectures.seq2seq import LSTMSeq2Seq, CNNLSTMSeq2Seq
from architectures.seq2point import LSTMSeq2Point, CNNLSTMSeq2Point
```
# Configuration
Supports $(B,L_{in},C_{in})\xrightarrow{network}(B,L_{out},C_{out})$ operations, where  
```math
\begin{aligned}
B&=\text{batch\_size}\\
L_{in}&=\text{input\_sequence\_length (variable)}\\
C_{in}&=\text{input\_embedding\_size}\\
L_{out}&=\text{output\_sequence\_length (variable)}\\
C_{out}&=\text{output\_embedding\_size}\\
\end{aligned}
```
- ```hidden_size``` Hidden state size of LSTM encoder.  
- ```num_layers``` Number of stacks in CNN, LSTM encoder, LSTM decoder, and FC layers.  
- ```bidirectional``` Whether to use bidirectional LSTM encoder.  
- ```dropout``` Dropout rate. Applies to:  
Residual drop path in 1DCNN  
hidden state dropout in LSTM encoder/decoder(for every time step).    
Unlike ```torch.nn.LSTM```, dropout is applied from the first LSTM layer.  
- ```layernorm``` Layer normalization in LSTM encoder and decoder.  
- ```attention``` Attention in LSTM decoder.  
Supports ```'bahdanau'``` for Bahdanau style, ```'dotproduct'``` for Scaled Dot Product style, and ```'none``` for non-attended decoder.

**All network parameters are initialized from $\mathcal{N}\sim(0,0.01^2)$, except for bias initialized from ```torch.zeros```. See ```architectures.init```**

```
B = 32  # batch size
Lin = 100  # input sequence length
Cin = 6  # input embedding size
Lout = 20  # output sequence length
Cout = 50  # output embedding size

hidden_size = 256
num_layers = 3
bidirectional = True 
dropout = 0.3
layernorm = True 
attention = "bahdanau"
```

# Create net instances and input/output

```
x = torch.randn(B, Lin, Cin)
y = torch.randn(B, Lout, Cout)

# LSTM encoder - LSTM decoder - MLP
seq2seq_lstm = LSTMSeq2Seq(
    Cin, Cout, hidden_size, num_layers, bidirectional, dropout, layernorm, attention
)  

# 1DCNN+LSTM encoder - LSTM decoder - MLP
seq2seq_cnnlstm = CNNLSTMSeq2Seq(
    Cin, Cout, hidden_size, num_layers, bidirectional, dropout, layernorm, attention
)

# LSTM encoder - MLP
seq2point_lstm = LSTMSeq2Point(
    Cin, Cout, hidden_size, num_layers, bidirectional, dropout, layernorm
)

# 1DCNN+LSTM encoder - MLP
seq2point_cnnlstm = CNNLSTMSeq2Point(
    Cin, Cout, hidden_size, num_layers, bidirectional, dropout, layernorm
)
```

# Forward operation
- ```x``` Input to the network. Supports $(B,L_{in},C_{in})$ only.  
- ```teacher_forcing``` Teacher forcing ratio $\in [0,1]$. Defaults to -1 (fully autoregressive).  
- ```y``` Output label for teacher forcing. Supports $(B,*,C_{out})$ only. Defaults to ```None``` (fully autoregressive).  
- ```trg_len``` Target sequence length to generate. Defaults to ```1```.
```
# Seq2Seq forward
outseq_lstm = seq2seq_lstm.forward(x, y, teacher_forcing=0.5, trg_len=Lout)
outseq_cnnlstm = seq2seq_cnnlstm.forward(x, y, teacher_forcing=0.5, trg_len=Lout)

# Seq2Point forward
outpoint_lstm = seq2point_lstm.forward(x)
outpoint_cnnlstm = seq2point_cnnlstm.forward(x)

print(outseq_lstm.shape)
print(outseq_cnnlstm.shape)
print(outpoint_lstm.shape)
print(outpoint_cnnlstm.shape)
```
```
torch.Size([32, 20, 50])
torch.Size([32, 20, 50])
torch.Size([32, 50])
torch.Size([32, 50])
```
# Accessing model properties
By inheriting ```architectures.skeleton.Skeleton```, model properties are automatically saved to attributes:  
- Parameters can be counted by ```model.count_params()```
- Properties are accessed using ```model.model_info``` attribute.  
- The identical model instance can be created by ```ModelClass(**model.model_init_args)```.  
```
seq2seq_lstm.count_params()

model_info = seq2seq_lstm.model_info
model_init_args = seq2seq_lstm.model_init_args
print(model_info)

another_model_instance = LSTMSeq2Seq(**model_init_args)
```
```
Number of trainable parameters: 10422835
{'attention': 'bahdanau', 'bidirectional': True, 'dropout': 0.3, 'hidden_size': 256, 'input_size': 6, 'layernorm': True, 'num_layers': 3, 'output_size': 50}
```
