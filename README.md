# TimeSeriesSeq2Seq
Seq2Seq, Seq2Point modeling implementations using 1D convolution, LSTM, Attention mechanisms, Transformer, and Temporal Fusion Transformer(TFT).  
The repo implements the following:  
- Basic convolution and LSTM layers implementation
- Bahdanau attention LSTM Encoder-Decoder network by [Bahdanau et al.(2014)](https://arxiv.org/abs/1409.0473)
- Vanilla Transformer by [Vaswani et al.(2017)](https://arxiv.org/abs/1706.03762). See ```architectures.transformer.Transformer```.
- Temporal Fusion Transformer by [Lim et al.(2020)](https://arxiv.org/abs/1912.09363). See See ```architectures.tft.TemporalFusionTransformer```.

**Transformer**-based classes always produce sequence-to-sequence outputs.  
**RNN**-based classes can selectively produce sequence or point outputs:  
- Difference between ```rnn_seq2seq``` and ```rnn_seq2point``` is the decoder part. The former uses autoregressive LSTM decoder to generate sequence of vectors, while the latter uses MLP decoder to generate a single vector.
 


**All network parameters are initialized from $\mathcal{N}\sim(0,0.01^2)$, except for bias initialized from ```torch.zeros```. See ```architectures.init```**.

See [Tutorial.ipynb](https://github.com/hyeonbeenlee/NeuralSeq2Seq/blob/main/Tutorial.ipynb) for details.  

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
Supports ```'bahdanau'``` for Bahdanau style, ```'dotproduct'``` for Dot Product style, and ```'none``` for non-attended decoder.


# Creating model instances

```
from architectures.rnn_seq2seq import *
from architectures.rnn_seq2point import *
from architectures.transformer import *
from architectures.tft import *

# LSTM encoder - LSTM decoder - MLP
seq2seq_lstm = LSTMSeq2Seq(
    Cin, Cout, hidden_size, num_layers, bidirectional, dropout, layernorm, attention
)  

# 1DCNN+LSTM encoder - LSTM decoder
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

# Transformer
seq2seq_transformer = Transformer(
    Cin, Cout, num_layers, n_heads, d_model, dropout, d_ff
)

# Temporal Fusion Transformer
seq2seq_tft = TemporalFusionTransformer(
    Cin, Cout, num_layers, n_heads, d_model, dropout
)
```

# Autoregressive forward operation
- ```x``` Input to the network. Supports $(B,L_{in},C_{in})$ only.  
- ```y``` Output label for teacher forcing. Supports $(B,*,C_{out})$ only. Defaults to ```None``` (fully autoregressive).
- ```teacher_forcing``` Teacher forcing ratio $\in [0,1]$. Defaults to ```-1``` (fully autoregressive).  
- ```trg_len``` Target sequence length to generate. Defaults to ```1```.

If only ```x``` and ```trg_len``` is given as arguments, the model will autoregressively produce ```trg_len``` length of outputs. 

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
