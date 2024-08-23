import pandas as pd
import numpy as np
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from torchsummary import summary


class BaseModel(nn.Module):
    def __init__(self, device, lr):
        super().__init__()
        self.device = device
        self.lr = lr
        self.train_losses = []

    def forward(self, x):
        pass

    def one_hot_encoding(self, y, num_classes=10):
        return torch.eye(num_classes)[y]

    def showTrainLoss(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('training_loss.png')
        #plt.show()


    
    def Train(self, dataloader, epoch):
        self.train()
        self.train_losses = []
        for e in range(0, epoch):
            epoch_loss = 0.0
            for i, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                label = self.one_hot_encoding(y).to(self.device)
                b, c, w, h = x.shape
                y_hat = self.forward(x)


                self.optimizer.zero_grad()
                loss = self.loss(y_hat, label)
                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()


                if i % 100 == 0:
                    print(f"[epoch : {e} , iteration : {i} / {len(dataloader)}] ..> train loss : {loss} ")

            avg_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_loss)

        self.showTrainLoss()
        return

    def Vaild(self, dataloader, num_classes=10):
        self.eval()

        correct = 0
        total = 0

        precision_metric = MulticlassPrecision(num_classes=num_classes, average='weighted').to(self.device)
        recall_metric = MulticlassRecall(num_classes=num_classes, average='weighted').to(self.device)
        f1_metric = MulticlassF1Score(num_classes=num_classes, average='weighted').to(self.device)

        total_iter = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                total_iter+=1
                x = x.to(self.device)
                y = y.to(self.device)


                y_hat = self.forward(x)
                predicted = torch.argmax(y_hat, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                precision_metric.update(predicted, y)
                recall_metric.update(predicted, y)
                f1_metric.update(predicted, y)


        accuracy = correct / total
        precision = precision_metric.compute()
        recall = recall_metric.compute()
        f1 = f1_metric.compute()
        
        print(f'Test Accuracy: {accuracy:.4f}')
        print(f'precision: {precision:.4f}')
        print(f'recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        return accuracy, f1
    

class AttentionLayer(nn.Module):
    def __init__(self,d_in, d_v, d_k = 100, device = 'cpu', lr = 0.0001):
        """

        :param token_length: input vector length
        :param d_in: input vector dimension
        :param d_v: value vector dimension
        :param d_k: key vector dimension
        :param device:
        :param lr:

        Attention layer get (token_length , d_in) vector and return (token_length, d_v) size Attention vector
        """
        super().__init__()

        self.d_k = d_k
        self.d_in = d_in
        self.d_v = d_v
        self.device = device
        self.learning_rate = lr
        self.q_net = nn.Linear(d_in,d_k) # Dimension -> [batch_size, token_length, input_vector_dimension ->[b, t, key_vector_dimension]
        self.k_net = nn.Linear( d_in, d_k)
        self.v_net = nn.Linear( d_in, d_v)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Attention
        :param x:
        :return:
        """
        b, t, i = x.shape

        q = self.q_net(x)
        k = self.k_net(x)
        v = self.v_net(x)

        # first, get Attention Distribution Q * K^(T)
        self.Attn_Dist = torch.matmul(q , k.transpose(-2, -1))

        # divide by Key vector dimention for stability

        self.Attn_Dist = self.Attn_Dist / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # get Attention Score
        self.Attn_Score = self.softmax(self.Attn_Dist)

        self.out = torch.matmul(self.Attn_Score , v)
        # output tensor size must be [token_length, d_v]
        return self.out

class MultiheadAttentionLayer(nn.Module):
    def __init__(self, token_length , d_in, d_v, d_k, d_out , head_n = 8, device = 'cpu', lr = 0.0001):
        """

        :param token_length: input vector length
        :param d_in: input vector dimension
        :param d_v: value vector dimension
        :param d_k: key vector dimension
        :param d_out: result vector after Linear layer dimension
        :param head_n: number of Attention layer head
        :param device:
        :param lr:

        MultiheadAttentionLayer get  (token_length, D_in) size vector and return (d_out) size Attention Embedded Vector
        """
        super().__init__()
        self.d_in = d_in
        self.d_v = d_v
        self.d_k = d_k
        self.token_length = token_length
        self.head_n = head_n
        self.device = device
        self.learning_rate = lr

        self.Attention_Heads = nn.ModuleList(AttentionLayer(self.d_in, self.d_v, self.d_k, self.device, self.learning_rate).to(self.device) for i in range(0 ,self.head_n))


        self.fc_layer = nn.Linear(self.token_length * self.d_v * self.head_n , d_out)

    def forward(self, x):
        result = []
        b, t, i = x.shape
        for i in range(0, self.head_n):
            result.append(self.Attention_Heads[i].forward(x))

        result = torch.concat(result, dim=1)
        result = result.view(b, -1)
        out = self.fc_layer(result)
        return out

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads


        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        #self.fc = nn.Linear(self.head_dim * self.seq_length * self.embed_dim, self.out_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size() # [batch_size, sequence_length, token_dimension]

        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim) # [batch_size, sequence_length, number of heads, token_dimension // number of heads -> vector dimension after heads
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs

        self.attn_dist = torch.matmul(q, torch.transpose(k, -2, -1)) / torch.sqrt(torch.tensor(self.head_dim)) # Transpose k : [batch_size, head_number, sequence_length, vector_dimension] -> [b, h, vector_dimension, seqence_length]
        self.attn_dist = torch.softmax(self.attn_dist, dim=3)

        self.attn = torch.matmul(self.attn_dist, v)

        values = self.attn.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim) # [batch, seqLen, flatten (Head* Dims) ]
        #out = self.o_proj(values)

        return values

class TransformerBlock(nn.Module):
    def __init__(self, token_length, d_in, head_n, device, lr):
        """

        :param token_length:
        :param d_in:
        :param head_n:
        :param device:
        :param lr:
        """
        super().__init__()
        self.token_length = token_length
        self.d_in = d_in
        self.head_n = head_n
        self.device = device
        self.learning_rate = lr
        self.AttentionOutShape = token_length * d_in
        self.multiheadAttn = MultiheadAttention(d_in, d_in, head_n)
        self.layerNorm = nn.LayerNorm([token_length,d_in])
        self.fc_layer = nn.Linear(d_in, 2048)
        self.relu = nn.ReLU()
        self.fc_layer2 = nn.Linear(2048, d_in)
        self.layerNorm2 = nn.LayerNorm([token_length, d_in])

    def forward(self, x):
        b, t, i = x.shape
        attn = self.multiheadAttn.forward(x) # output tensor size must be [batch, token_length * d_in]
        out1 = self.layerNorm(attn + x)
        out2 = self.relu(self.fc_layer(out1))
        out2 = self.fc_layer2(out2)
        result = self.layerNorm2(out1 + out2)
        return result

class VITBlock(TransformerBlock):
    def __init__(self, token_length, d_in, head_n, device, lr):
        super().__init__(token_length, d_in, head_n, device, lr)
        self.layerNorm = nn.LayerNorm([token_length+1,d_in])
    def forward(self, x):
        b,t,i = x.shape
        out = self.layerNorm(x)
        out = self.multiheadAttn.forward(out)
        out_multiAttn = out + x # skip connection
        out = self.layerNorm(out_multiAttn)
        out2 = self.fc_layer(out)
        out2 = self.relu(self.fc_layer(out))
        out2 = self.fc_layer2(out2)
        return out2 + out_multiAttn


class VisionTransformer(BaseModel):
    def __init__(self,  image_size, patch_size, d_in, head_n = 8, device = "cpu", lr = 0.0001):
        """

        :param image_size:
        :param patch_size:
        :param d_in:
        :param head_n:
        :param device:
        :param lr:
        """
        super().__init__(device, lr)
        self.d_in = d_in
        self.head_n = head_n
        self.token_length = (image_size // patch_size) ** 2

        self.patchEmb = PatchEmbedding(image_size, patch_size, 3, d_in)
        self.posEnc = PositionalEncoding(d_in, self.token_length+1)
        self.transformer = nn.ModuleList(VITBlock(self.token_length, self.d_in,head_n,device, lr) for i in range(0, 2))
        #self.transformer = nn.ModuleList(VITBlock(self.token_length, self.d_in, head_n, device, lr) for i in range(0, 16))
        self.fc1 = nn.Linear((self.token_length + 1) * d_in, self.token_length * d_in // 2)
        self.norm = nn.BatchNorm1d(self.token_length * d_in // 2)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(self.token_length * d_in // 2 , 10)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(0.2)
        #self.crossEnt = 
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.class_token = nn.Parameter(torch.randn(1, 1, d_in))
    def forward(self, x):
        b, c, w, h = x.shape
        out = self.patchEmb.forward(x) # image to patchEmbvec : [b, c, w, h] -> [b , patch_number {(image_size / patch_size)**2} = token_length , d_in]
        repeat_cls = self.class_token.repeat(x.shape[0], 1, 1)
        out = torch.concat((repeat_cls, out), dim=1)
        out = self.posEnc.forward(out)
        for i in range(0,2):
            out = self.dropout(out)
            out = self.transformer[i].forward(out)
        #out = self.transformer.forward(out) # [b , token_length, d_in]
        b, t, i = out.shape
        out = out.view(b, -1)

        out = self.silu(self.fc1.forward(out))
        out = self.fc2.forward(out)
        #out = self.softmax(self.fc2.forward(out))

        return out



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_dim = in_channels * patch_size * patch_size

        # Linear layer to project flattened patches to embedding dimension
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # Divide image into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, self.num_patches, -1)

        # Project patches to embedding dimension
        embeddings = self.proj(patches)
        return embeddings
class PatchEmb(nn.Module):
    def __init__(self, patch_size, channel, Emb_d):
        super().__init__()
        self.patch_size = patch_size
        self.channel = channel
        self.vector_size = patch_size * patch_size * channel
        self.Emb_d = Emb_d
        self.Conv = nn.Conv2d(channel, self.vector_size, patch_size,  patch_size) # PatchEmbedding process do just flat the patch but, in VIT paper, using Conv2d has better performance.

        self.Proj = nn.Linear(self.vector_size, self.Emb_d)
        pass

    def forward(self,x):
        b, c, w, h = x.shape
        output = self.Conv(x).view(b, self.vector_size, -1) # [b, c, w, d] -> [b, c * patch_size * patch_size, image_size / patch_size]
        output = torch.transpose(output, 2, 1) # [b, number of patches , c * patch_size * patch_size]

        Emb_output = self.Proj(output) # vector size [b, number of pathces, vector_size] -> [b, number of pathces, embedding vector size ]

        return Emb_output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class SimpleCNN_L(BaseModel):
    def __init__(self, imageSize, outDim, device, lr):
        super().__init__(device, lr)

        self.Conv1 = nn.Conv2d(3, 32, 7, 2, ) # [b, c, w, h] -> [b, c, w/2 - 3, h/2 - 3]
        self.Conv2 = nn.Conv2d(32, 64, 6, 1) # [b, c, w//2 -8, h/2 -8]
        self.norm = nn.LayerNorm([imageSize//2 -8, imageSize//2 - 8])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2) # [b, c, w/4-4, h/4 -4]


        self.Conv3 = nn.Conv2d(64, 96, 5, 1) # [b, c, w/4 - 8, h/4 - 8]
        self.norm2 = nn.LayerNorm([imageSize//4 -8, imageSize//4 -8]) # [b/ c/ w/8-4, h/8-4]
        #self.Conv4 = nn.Conv2d(32, 64, 5,  1) # [b, c, w/4 -14, h/4 -14] -> #b, c, w/8 -7 , h/8 -7
        """self.Conv1 = nn.Conv2d(3, 32, 3, 1, 1) # [ b, c, w, h ] -> [b, c, w, h] 
        self.norm = nn.LayerNorm([imageSize, imageSize])
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # [b, c, w, h] -> [b, c, w/2, h/2]
        self.Conv2 = nn.Conv2d(32, 64, 3, 1, 1)"""

        self.Linear = nn.Linear(96 * (imageSize//8 - 4) * (imageSize//8 - 4) , (imageSize//8 -4))

        #self.Linear2 = nn.Linear((imageSize//8 -4)**2, (imageSize//8 -4))
        self.Linear3 = nn.Linear((imageSize//8 -4) , outDim)
        self.softmax = nn.Softmax(dim=1)

        self.loss = nn.CrossEntropyLoss()
        #self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        
    def forward(self,x):
        b, c, w, h = x.shape

        out = self.Conv1(x)
        out = self.Conv2(out)
        #out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Conv3(out)
        #out = self.Conv4(out)
        #out = self.norm2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.relu(out)
        out = out.view(b, -1)
        out = self.relu(self.Linear(out))
        #out = self.relu(self.Linear2(out))
        out = self.Linear3(out)
        #out = self.softmax(out)
        return out

class SimpleCNN_S(SimpleCNN_L):
    def __init__(self, imageSize, outDim, device, lr):
        super().__init__(imageSize, outDim, device, lr)
        self.Conv1 = nn.Conv2d(3, 16, 5, 1) # [b, c, w-4, h-4]
        self.Conv2 = nn.Conv2d(16, 32, 5, 1) # [b, c, w-8, h-8]

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.Conv3 = nn.Conv2d(32, 64, 3, 1) #[b, c, w//2 -4 -2 , h//2 -4 -2]


        self.linear = nn.Sequential(
            nn.Linear(64 * (imageSize//2 - 6)**2 , 128),
            nn.ReLU(),
            nn.Linear(128, outDim),
        )
        self.loss = nn.CrossEntropyLoss()
        #self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

    def forward(self,x):
        b, c, w, h = x.shape
        out = self.Conv1(x)
        out = self.Conv2(out)

        out = self.relu(out)
        out = self.maxpool(out)
        out = self.Conv3(out)

        out = out.view(b, -1)
        out = self.linear(out)
        return out

# Define a simple CNN model
class SimpleCNN(SimpleCNN_L):
    def __init__(self, imageSize, outDim, device, lr):
        super().__init__(imageSize, outDim, device, lr)
        # First convolutional layer with 32 filters, 3x3 kernel, and ReLU activation
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Second convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Third convolutional layer with 64 filters, 3x3 kernel, and ReLU activation
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # 64 filters * 4x4 feature maps
        self.fc2 = nn.Linear(128, 10)  # 10 output classes

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(b , -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


        
        




