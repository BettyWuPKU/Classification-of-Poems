# %%
!nvidia-smi 

# %%
from google.colab import drive
import os
drive.mount('/content/drive')

print(os.getcwd())
os.chdir("/content/drive/My Drive/Colab Notebooks")

# 返回上一级目录

# %% [markdown]
# # RNN序列编码-分类期末大作业
# 
# 本次大作业要求手动实现双向LSTM+基于attention的聚合模型，并用于古诗作者预测的序列分类任务。**请先阅读ppt中的作业说明。**

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from tqdm import tqdm

device = torch.device("cuda:0")
print(torch.cuda.is_available())

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# %% [markdown]
# ## 1. 加载数据

# %% [markdown]
# 数据位于`data`文件夹中，每一行对应一个样例，格式为“诗句 作者”。下面的代码将数据文件读取到`train_data`, `valid_data`和`test_data`中，并根据训练集中的数据构造词表`word2idx`/`idx2word`和标签集合`label2idx`/`idx2label`。

# %%
word2idx = {"<unk>": 0}
label2idx = {}
idx2word = ["<unk>"]
idx2label = []

train_data = []
with open("data/train.txt") as f:
    for line in f:
        # strip移除头尾的空格
        text, author = line.strip().split()
        for c in text:
            if c not in word2idx:
                word2idx[c] = len(idx2word)
                idx2word.append(c)
        if author not in label2idx:
            label2idx[author] = len(idx2label)
            idx2label.append(author)
        train_data.append((text, author))

valid_data = []
with open("data/valid.txt") as f:
    for line in f:
        text, author = line.strip().split()
        valid_data.append((text, author))

test_data = []
with open("data/test.txt") as f:
    for line in f:
        text, author = line.strip().split()
        test_data.append((text, author))

# %%
print(len(word2idx), len(idx2word), len(label2idx), len(idx2label))
print(len(train_data), len(valid_data), len(test_data))

# %% [markdown]
# **请完成下面的函数，其功能为给定一句古诗和一个作者，构造RNN的输入。** 这里需要用到上面构造的词表和标签集合，对于不在词表中的字用\<unk\>代替。

# %%
def make_data(text, author):
    """
    输入
        text: str
        author: str
    输出
        x: LongTensor, shape = (1, text_length)
        y: LongTensor, shape = (1,)
    """
    x = torch.LongTensor([[word2idx.get(c) if word2idx.get(c) != None else 0 for c in text]])
    y = torch.LongTensor([label2idx.get(author)])
    return x, y

# %% [markdown]
# ## 2. LSTM算子（单个时间片作为输入）

# %%
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i = nn.Linear(input_size + hidden_size, hidden_size)
        self.o = nn.Linear(input_size + hidden_size, hidden_size)
        self.g = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, ht, ct, xt):
        # ht: 1 * hidden_size
        # ct: 1 * hidden_size
        # xt: 1 * input_size
        input_combined = torch.cat((xt, ht), 1)
        ft = torch.sigmoid(self.f(input_combined))
        it = torch.sigmoid(self.i(input_combined))
        ot = torch.sigmoid(self.o(input_combined))
        gt = torch.tanh(self.g(input_combined))
        ct = ft * ct + it * gt
        ht = ot * torch.tanh(ct)
        return ht, ct

# %% [markdown]
# ## 3. 实现双向LSTM（整个序列作为输入）
# 
# **要求使用上面提供的LSTM算子，不要调用torch.nn.LSTM**

# %%
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiLSTM, self).__init__()
        # TODO
        # input_size就是embedding的维数？
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_for = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.lstm_back = LSTM(input_size=input_size, hidden_size=hidden_size)
        self.register_buffer("_float", torch.zeros(1, hidden_size))
    
    def init_h_and_c(self):
        h = torch.zeros_like(self._float)
        c = torch.zeros_like(self._float)
        return h, c
    
    def forward(self, x):
        """
        输入
            x: 1 * length * input_size
        输出
            hiddens
        """
        # TODO
        input_len = x.shape[1]
        hidden_output_for = []
        hidden_output_back = []
        h_for, c_for = self.init_h_and_c()
        h_back, c_back = self.init_h_and_c()
        for i in range(input_len):
            h_for, c_for = self.lstm_for(h_for, c_for, x[0][i].unsqueeze(0))
            h_back, c_back = self.lstm_back(h_back, c_back, x[0][input_len-i-1].unsqueeze(0))
            hidden_output_for.append(h_for)
            hidden_output_back.append(h_back)
        hidden_output_back.reverse()
        
        hiddens = torch.cat([torch.stack(hidden_output_for), torch.stack(hidden_output_back)], dim=-1)
        
        return hiddens.transpose(1,0)

# %% [markdown]
# ## 4. 实现基于attention的聚合机制

# %%
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # TODO
        self.q = torch.randn((hidden_size,1), requires_grad=True).cuda()
        self.lin = nn.Linear(hidden_size, hidden_size, bias=False)
        
    
    def forward(self, hiddens):
        """
        输入
            hiddens: 1 * length * hidden_size
        输出
            attn_outputs: 1 * hidden_size
        """
        # TODO
        out = torch.bmm(hiddens, self.q.unsqueeze(0))
        out = F.softmax(out, dim=1) 
        attn_outputs = (out * hiddens).sum(dim=1)
        return attn_outputs

# %% [markdown]
# ## 5. 利用上述模块搭建序列分类模型
# 
# 参考模型结构：Embedding – BiLSTM – Attention – Linear – LogSoftmax

# %%
class EncoderRNN(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_size, num_classes):
        """
        参数
            num_vocab: 词表大小
            embedding_dim: 词向量维数
            hidden_size: 隐状态维数
            num_classes: 类别数量
        """
        super(EncoderRNN, self).__init__()
        # TODO
        self.num_vocab = num_vocab
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.emb = nn.Embedding(num_vocab, embedding_dim)
        self.biLSTM = BiLSTM(embedding_dim, hidden_size)
        self.attn = Attention(2*hidden_size)
        self.lin = nn.Linear(2*hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        """
        输入
            x: 1 * length, LongTensor
        输出
            outputs
        """
        # TODO
        embedding = self.emb(x)
        bi_lstm_output = self.biLSTM(embedding)
        attn_output = self.attn(bi_lstm_output)
        # out num_classes
        output = self.softmax(self.lin(attn_output))
        return output

# %% [markdown]
# ## 6. 请利用上述模型在古诗作者分类任务上进行训练和测试
# 
# 要求选取在验证集上效果最好的模型，输出测试集上的准确率、confusion matrix以及macro-precision/recall/F1，并打印部分测试样例及预测结果。

# %%
# TODO
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels_name, title):
    plt.title(title)
    sns.heatmap(cm, cmap='Blues', annot=True, fmt='g')
    plt.show()


def collate(datalist):
    l = [make_data(text, author) for text, author in datalist]
    return [s[0] for s in l], [s[1] for s in l]
    
batch_size = 16
trainloader = torch.utils.data.DataLoader(train_data, 
            batch_size=batch_size, shuffle=True, collate_fn=collate)
validloader = torch.utils.data.DataLoader(valid_data, 
            batch_size=batch_size, shuffle=True, collate_fn=collate)
testloader = torch.utils.data.DataLoader(test_data, 
            batch_size=batch_size, shuffle=True, collate_fn=collate)

def train_loop(model, optimizer, criterion, loader):
    model.train()
    epoch_loss = 0.0
    for src, tgt in tqdm(loader):
        B = len(src)
        loss = 0.0
        for _ in range(B):
            _src = src[_].to(device)
            _tgt = tgt[_].to(device)
            outputs = model(_src)

            loss += criterion(outputs, _tgt)

        loss /= B
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    return epoch_loss

def test_loop(model, loader):
    model.eval()
    correct = 0.0
    total = 0
    conf_matrix = torch.zeros([len(label2idx), len(label2idx)])
    y_pred = []
    y_true = []

    for src, tgt in tqdm(loader):
        B = len(src)
        for _ in range(B):
            _src = src[_].to(device)
            with torch.no_grad():
                outputs = model(_src)
            prediction = torch.max(outputs, 1)[1]
            correct += (prediction.item() == tgt[_]).float()
            y_pred.append(prediction.item())
            y_true.append(tgt[_].item())
            conf_matrix[tgt[_].item()][prediction.item()] += 1
        total += len(tgt)

    accuracy = correct / total
    f1 = f1_score(y_true, y_pred, average='macro')
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    plot_confusion_matrix(conf_matrix, label2idx.keys, "Confusion Matrix")
    return (accuracy, f1, p, r)
    


# %%
torch.manual_seed(1)
model = EncoderRNN(num_vocab=len(word2idx), embedding_dim=256, hidden_size=256, num_classes=len(label2idx))
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)
criterion = nn.NLLLoss()

best_score = 0.0
for _ in range(15):
    loss = train_loop(model, optimizer, criterion, trainloader)
    (accuracy, f1, p, r) = test_loop(model, validloader)
    # 保存验证集上accuracy最高的checkpoint
    if accuracy > best_score:
        torch.save(model.state_dict(), "model_best.pt")
        best_score = accuracy
    print(f"Epoch {_}: loss = {loss}, accuracy = {accuracy}, macro F1 = {f1}, macro precision = {p}, macro recall = {r}")

# %%
model.load_state_dict(torch.load("model_best.pt"))
accuracy, f1, p, r = test_loop(model, testloader)
print(f"Test accuracy = {accuracy}, macro F1 = {f1}, macro precision = {p}, macro recall = {r}")

# %%
for t in range(0,100,10):
  text_data, author_data = make_data(test_data[t][0], test_data[t][1])
  text_data = text_data.cuda()
  with torch.no_grad():
      outputs = model(text_data)
  prediction = torch.max(outputs, 1)[1].item()
  print(f"predicting the author of {test_data[t][0]}")
  print(f"The predicted author: {idx2label[prediction]}, real author: {test_data[t][1]}")

# %%



