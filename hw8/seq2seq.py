import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import sys
import os
import random
import json

device = torch.device("cuda")# if torch.cuda.is_available() else "cpu") # 判斷是用 CPU 還是 GPU 執行運算

import numpy as np

class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label

"""## 定義 Dataset
- Data (出自manythings 的 cmn-eng):
    - 訓練資料：18000句
    - 檢驗資料：    500句
    - 測試資料： 2636句

- 資料預處理:
    - 英文：
        - 用 subword-nmt 套件將word轉為subword
        - 建立字典：取出標籤中出現頻率高於定值的subword
    - 中文：
        - 用 jieba 將中文句子斷詞
        - 建立字典：取出標籤中出現頻率高於定值的詞
    - 特殊字元： < PAD >, < BOS >, < EOS >, < UNK > 
        - < PAD >    ：無意義，將句子拓展到相同長度
        - < BOS >    ：Begin of sentence, 開始字元
        - < EOS >    ：End of sentence, 結尾字元
        - < UNK > ：單字沒有出現在字典裡的字
    - 將字典裡每個 subword (詞) 用一個整數表示，分為英文和中文的字典，方便之後轉為 one-hot vector     

- 處理後的檔案:
    - 字典：
        - int2word_*.json: 將整數轉為文字
        ![int2word_en.json](https://i.imgur.com/31E4MdZ.png)
        - word2int_*.json: 將文字轉為整數
        ![word2int_en.json](https://i.imgur.com/9vI4AS1.png)
        - $*$ 分為英文（en）和中文（cn）
    
    - 訓練資料:
        - 不同語言的句子用 TAB ('\t') 分開
        - 字跟字之間用空白分開
        ![data](https://i.imgur.com/nSH1fH4.png)


- 在將答案傳出去前，在答案開頭加入 "< BOS >" 符號，並於答案結尾加入 "< EOS >" 符號
"""

import re
import json

class EN2CNDataset(data.Dataset):
    def __init__(self, root, max_output_len, set_name):
        self.root = root

        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        # 載入資料
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), "r") as f:
            for line in f:
                self.data.append(line)
        print (f'{set_name} dataset size: {len(self.data)}')

        self.cn_vocab_size = len(self.word2int_cn)
        self.en_vocab_size = len(self.word2int_en)
        self.transform = LabelTransform(max_output_len, self.word2int_en['<PAD>'])
        self.max_output_len = max_output_len

    def get_dictionary(self, language):
        # 載入字典
        with open(os.path.join(self.root, f'word2int_{language}.json'), "r") as f:
            word2int = json.load(f)
        with open(os.path.join(self.root, f'int2word_{language}.json'), "r") as f:
            int2word = json.load(f)
        return word2int, int2word

    def __len__(self):
        return len(self.data)

    def __getitem__(self, Index):
        # 先將中英文分開
        sentences = self.data[Index]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))
        #print (sentences)
        assert len(sentences) == 2

        # 預備特殊字元
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # 在開頭添加 <BOS>，在結尾添加 <EOS> ，不在字典的 subword (詞) 用 <UNK> 取代
        en, cn = [BOS], [BOS]
        # 將句子拆解為 subword 並轉為整數
        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        #print (f'en: {sentence}')
        for word in sentence[:self.max_output_len-2]:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        # 將句子拆解為單詞並轉為整數
        # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
        sentence = re.split(' ', sentences[1])
        sentence = list(filter(None, sentence))
        #print (f'cn: {sentence}')
        for word in sentence[:self.max_output_len-2]:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.asarray(en), np.asarray(cn)

        # 用 <PAD> 將句子補到相同長度
        en, cn = self.transform(en), self.transform(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        return en, cn

"""# 模型架構

## Encoder
- seq2seq模型的編碼器為RNN。 對於每個輸入，，**Encoder** 會輸出**一個向量**和**一個隱藏狀態(hidden state)**，並將隱藏狀態用於下一個輸入，換句話說，**Encoder** 會逐步讀取輸入序列，並輸出單個矢量（最終隱藏狀態）
- 參數:
    - en_vocab_size 是英文字典的大小，也就是英文的 subword 的個數
    - emb_dim 是 embedding 的維度，主要將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
    - hid_dim 是 RNN 輸出和隱藏狀態的維度
    - n_layers 是 RNN 要疊多少層
    - dropout 是決定有多少的機率會將某個節點變為 0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不使用
- Encoder 的輸入和輸出:
    - 輸入: 
        - 英文的整數序列 e.g. 1, 28, 29, 205, 2
    - 輸出: 
        - outputs: 最上層 RNN 全部的輸出，可以用 Attention 再進行處理
        - hidden: 每層最後的隱藏狀態，將傳遞到 Decoder 進行解碼
"""

class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch size, sequence len, vocab size]
        embedding = self.embedding(input)
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =    [num_layers * directions, batch size    , hid dim]
        # outputs 是最上層RNN的輸出
                
        return outputs, hidden

"""## Decoder
- **Decoder** 是另一個 RNN，在最簡單的 seq2seq decoder 中，僅使用 **Encoder** 每一層最後的隱藏狀態來進行解碼，而這最後的隱藏狀態有時被稱為 “content vector”，因為可以想像它對整個前文序列進行編碼， 此 “content vector” 用作 **Decoder** 的**初始**隱藏狀態， 而 **Encoder** 的輸出通常用於 Attention Mechanism
- 參數
    - en_vocab_size 是英文字典的大小，也就是英文的 subword 的個數
    - emb_dim 是 embedding 的維度，是用來將 one-hot vector 的單詞向量壓縮到指定的維度，主要是為了降維和濃縮資訊的功用，可以使用預先訓練好的 word embedding，如 Glove 和 word2vector
    - hid_dim 是 RNN 輸出和隱藏狀態的維度
    - output_dim 是最終輸出的維度，一般來說是將 hid_dim 轉到 one-hot vector 的單詞向量
    - n_layers 是 RNN 要疊多少層
    - dropout 是決定有多少的機率會將某個節點變為0，主要是為了防止 overfitting ，一般來說是在訓練時使用，測試時則不用
    - isatt 是來決定是否使用 Attention Mechanism

- Decoder 的輸入和輸出:
    - 輸入:
        - 前一次解碼出來的單詞的整數表示
    - 輸出:
        - hidden: 根據輸入和前一次的隱藏狀態，現在的隱藏狀態更新的結果
        - output: 每個字有多少機率是這次解碼的結果
"""

class Decoder(nn.Module):
    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, config.emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        # e.g. Attention 接在輸入後面會使得維度變化，所以輸入維度改為
        self.input_dim = emb_dim + hid_dim * 2# if isatt else emb_dim
        self.rnn = nn.GRU(self.input_dim, self.hid_dim, self.n_layers, dropout = dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size, vocab size]
        # hidden = [batch size, n layers * directions, hid dim]
        # Decoder 只會是單向，所以 directions=1
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        if self.isatt:
            attn = self.attention(encoder_outputs, hidden)
            embedded = torch.cat([embedded, attn], dim=-1)
        else:
            embedded = torch.cat([embedded, encoder_outputs[:, -1:]], dim=-1)
            # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]

        # 將 RNN 的輸出轉為每個詞出現的機率
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden

"""## Attention
- 當輸入過長，或是單獨靠 “content vector” 無法取得整個輸入的意思時，用 Attention Mechanism 來提供 **Decoder** 更多的資訊
- 主要是根據現在 **Decoder hidden state** ，去計算在 **Encoder outputs** 中，那些與其有較高的關係，根據關系的數值來決定該傳給 **Decoder** 那些額外資訊 
- 常見 Attention 的實作是用 Neural Network / Dot Product 來算 **Decoder hidden state** 和 **Encoder outputs** 之間的關係，再對所有算出來的數值做 **softmax** ，最後根據過完 **softmax** 的值對 **Encoder outputs** 做 **weight sum**

- TODO:
實作 Attention Mechanism
"""

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim
        units = 10
        self.w1 = nn.Linear(self.hid_dim*2, units, bias=False) # for encoder_outputs
        self.w2 = nn.Linear(self.hid_dim, units, bias=False)  # for decoder_hidden
        self.w3 = nn.Linear(units, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般來說是取 Encoder 最後一層的 hidden state 來做 attention
        encoder_out = self.w1(encoder_outputs)
        decoder_hid = self.w2(torch.unsqueeze(decoder_hidden[-1], 1))
        added_tanh = F.tanh(encoder_out + decoder_hid)
        score = self.w3(added_tanh)
        alpha = F.softmax(score) # [batch, sequence len, 1]
        attention = torch.bmm(torch.transpose(encoder_out, 1, 2), alpha)
        
        return attention # [batch, 1, hid_dim * directions]

"""## Seq2Seq
- 由 **Encoder** 和 **Decoder** 組成
- 接收輸入並傳給 **Encoder** 
- 將 **Encoder** 的輸出傳給 **Decoder**
- 不斷地將 **Decoder** 的輸出傳回 **Decoder** ，進行解碼    
- 當解碼完成後，將 **Decoder** 的輸出傳回
"""

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
                        "Encoder and decoder must have equal number of layers!"
                        
    def forward(self, input, target, teacher_forcing_ratio):
        # input    = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 是有多少機率使用正確答案來訓練
        batch_size = target.shape[0]
        target_len = target.shape[1]
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =    [num_layers * directions, batch size    , hid dim]    --> [num_layers, directions, batch size    , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            # 決定是否用正確答案來做訓練
            teacher_force = random.random() < teacher_forcing_ratio
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            # 如果是 teacher force 則用正解訓練，反之用自己預測的單詞做預測
            input = target[:, t] if teacher_force and t < target_len else top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

    def inference(self, input, target, beam_search=True):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1    
        # input    = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]                # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =    [num_layers * directions, batch size    , hid dim]    --> [num_layers, directions, batch size    , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds

"""# utils
- 基本操作:
    - 儲存模型
    - 載入模型
    - 建構模型
    - 將一連串的數字還原回句子
    - 計算 BLEU score
    - 迭代 dataloader

## 儲存模型
"""

def save_model(model, optimizer, store_model_path, step):
    os.makedirs(store_model_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(store_model_path, 'model.ckpt'))
    return

"""## 載入模型"""

def load_model(model, load_model_path):
    print(f'\033[32;1mLoad model from {load_model_path}\033[0m')
    model.load_state_dict(torch.load(os.path.join(load_model_path, 'model.ckpt')))
    return model

"""## 建構模型"""

def build_model(config, en_vocab_size, cn_vocab_size):
    # 建構模型
    encoder = Encoder(en_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
    decoder = Decoder(cn_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.attention)
    model = Seq2Seq(encoder, decoder, device)
    print(model)
    # 建構 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(optimizer)
    if config.load_model:
        model = load_model(model, config.load_model_path)
    model = model.to(device)

    return model, optimizer

"""## 數字轉句子"""

def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    
    return sentences

"""## 計算 BLEU score"""

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def computebleu(sentences, targets):
    score = 0 
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp 

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))                                                                                                                                                                                    
    
    return score

"""##迭代 dataloader"""

def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)

"""## schedule_sampling"""

def schedule_sampling(step, teacher_forcing_ratio):
    return teacher_forcing_ratio

"""# 訓練步驟

## 訓練
- 訓練階段
"""

def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset, teacher_forcing_ratio):
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        sources, targets = next(train_iter)
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling(step, teacher_forcing_ratio))
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print (f'train [{total_steps + step + 1}] loss: {loss_sum:.3f}, Perplexity: {np.exp(loss_sum):.3f}', end='\r')
            losses.append(loss_sum)
            loss_sum = 0.0
    print('')
    return model, optimizer, np.mean(losses)

"""## 檢驗/測試
- 防止訓練發生overfitting
"""

def test(model, dataloader, loss_function, beam_search):
    model.eval()
    loss_sum, bleu_score= 0.0, 0.0
    result = []
    for sources, targets in dataloader:
        sources, targets = sources.to(device), targets.to(device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets, beam_search)
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)

        loss = loss_function(outputs, targets)
        loss_sum += loss.item()

        # 將預測結果轉為文字
        targets = targets.view(sources.size(0), -1)
        preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
        sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
        targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        # 計算 Bleu Score
        bleu_score += computebleu(preds, targets)

    return loss_sum / len(dataloader), bleu_score / len(dataloader), result

"""## 訓練流程
- 先訓練，再檢驗
"""

def train_process(config):
    # 準備訓練資料
    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 準備檢驗資料
    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=1)
    # 建構模型
    model, optimizer = build_model(config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while total_steps < config.num_steps:
        # 訓練模型
        model, optimizer, loss = train(model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset, config.teacher_forcing_ratio)
        train_losses.append(loss)
        # 檢驗模型
        val_loss, bleu_score, result = test(model, val_loader, loss_function, config.beam_search)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)

        total_steps += config.summary_steps
        print (f'val [{total_steps}/{config.num_steps}] loss: {val_loss:.3f}, Perplexity: {np.exp(val_loss):.3f}, blue score: {bleu_score:.3f}')
        
        # 儲存模型和結果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, optimizer, config.store_model_path, total_steps)
            with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
                for line in result:
                    print (line, file=f)
        
    return train_losses, val_losses, bleu_scores

"""## 測試流程"""

def test_process(config):
    # 準備測試資料
    test_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'testing')
    test_loader = data.DataLoader(test_dataset, batch_size=1)
    # 建構模型
    model, optimizer = build_model(config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
    print("\033[32;1mFinish build model\033[0m")
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    # 測試模型
    test_loss, bleu_score, result = test(model, test_loader, loss_function, config.beam_search)
    # 儲存結果
    if config.output_file:
        with open(config.output_file, 'w') as f:
            for line in result:
                print (line, file=f)

    return test_loss, bleu_score

"""# Config
- 實驗的參數設定表
"""

class configurations(object):
    def __init__(self, data_dir, model_path, with_attention, teacher_forcing_ratio, beam_search, output_file, no_training):
        self.batch_size = 256
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.3
        self.learning_rate = 0.001
        self.max_output_len = 32+2  # 最後輸出句子的最大長度
        self.num_steps = 50 * 18000 // self.batch_size  # 總訓練次數
        self.store_steps = 2 * 18000 // self.batch_size  # 訓練多少次後須儲存模型
        self.summary_steps = 2 * 18000 // self.batch_size  # 訓練多少次後須檢驗是否有overfitting
        self.load_model = no_training  # 是否需載入模型
        self.store_model_path = model_path  # 儲存模型的位置
        self.load_model_path = model_path  # 載入模型的位置 e.g. "./ckpt/model_{step}" 
        self.data_path = data_dir  # 資料存放的位置
        self.attention = with_attention  # 是否使用 Attention Mechanism
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.beam_search = beam_search
        self.output_file = output_file

"""#Main Function
- 讀入參數
- 進行訓練或是推論

## train
"""
import argparse
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir')
    parser.add_argument('model_path')
    parser.add_argument('-T', '--no-training', action='store_true')
    parser.add_argument('-s', '--test', type=str, help='predicted file')
    parser.add_argument('-g', '--gpu', type=str, default='3')
    parser.add_argument('-t', '--teacher-forcing-ratio', type=float, default=1.0)
    parser.add_argument('-b', '--beam-search', type=int, default=False, help='Number of branches to do beam search. Disabled by default.')
    parser.add_argument('-a', '--attention', action='store_true')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config = configurations(args.data_dir, args.model_path, args.attention, args.teacher_forcing_ratio,
        args.beam_search, args.test, args.no_training)
    config.store_model_path = config.load_model_path = args.model_path
    print('config:\n', vars(config))
    if not args.no_training:
        train_losses, val_losses, bleu_scores = train_process(config)
        df = pd.DataFrame(list(zip(range(len(train_losses)), train_losses, val_losses, bleu_scores)),
            columns=['steps', 'train_loss', 'val_loss', 'bleu_score'])
        df.to_csv(os.path.join(config.store_model_path, 'logs.csv'), index=False)
    else:
        test_loss, bleu_score = test_process(config)
        print(f'\033[32;1mtest loss: {test_loss}, bleu_score: {bleu_score}\033[0m')
