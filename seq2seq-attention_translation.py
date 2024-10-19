# -*- coding: utf-8 -*-
# @Time    : 2022/11/11 18:50
# @Author  : 楚楚
# @File    : 01translation.py
# @Software: PyCharm

# !wget http://www.manythings.org/anki/cmn-eng.zip
# !unzip -d ./cmn-eng cmn-eng.zip

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import time
import math
import random
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
read()：
    读取整个文件，返回的是一个字符串，字符串包括文件中的所有内容
    若想要将每一行数据分离，即需要对每一行数据进行操作，此方法无效
    若内存不足无法使用此方法
readline()：
    每次读取下一行文件
    可将每一行数据分离
    主要是用场景：当内存不足时，使用readline()可以每次读取一行数据，只需要很少的内存
readlines()：
    一次读取所有行文件
    可将每一行数据分离，若需要对每一行数据进行处理，可以对readlines()求得的结果进行遍历
    若内存不足无法使用此方法
'''

# 读取数据
with open('./dataset/cmn.txt', 'r', encoding='utf-8') as file:
    data = file.read()
data = data.strip()
data = data.split('\n')

print(f"样本数：{len(data)}")
print(f'样本实例：{data[0]}')

# 分割英文数据和中文数据
english_data = [line.split('\t')[0] for line in data]
chinese_data = [line.split('\t')[1] for line in data]

print(f"english_data: {english_data[:10]}")
print(f"chinese_data: {chinese_data[:10]}")

# 按字符级切割，并添加<eos>
english_token_list = [[char for char in line] + ['<eos>'] for line in english_data]
chinese_token_list = [[char for char in line] + ['<eos>'] for line in chinese_data]

print(f"chinese_token_list: {chinese_token_list[:2]}")
print(f"english_token_list: {english_token_list[:2]}")

# 基本字典
basic_dictionary = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}

# 生成英文字典
'''
join():
    Example: '.'.join(['ab', 'pq', 'rs']) -> 'ab.pq.rs'

D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
    If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
'''
english_vocabulary = set(''.join(english_data))
print(f"english_vocabulary: {english_vocabulary}")
english2id = {char: i + len(basic_dictionary) for i, char in enumerate(english_vocabulary)}
english2id.update(basic_dictionary)
id2english = {v: k for k, v in english2id.items()}

print(f'english2id: {english2id}')
print(f'id2english: {id2english}')

# 生成中文字典
chinese_vocabulary = set(''.join(chinese_data))
print(f'chinese_vocabulary: {chinese_vocabulary}')
chinese2id = {char: i + len(basic_dictionary) for i, char in enumerate(chinese_vocabulary)}
chinese2id.update(basic_dictionary)
id2chinese = {v: k for k, v in chinese2id.items()}

print(f'chinese2id: {chinese2id}')
print(f'id2chinese: {id2chinese}')

# 利用字典，映射数据
english_reflect_data = [[english2id[en] for en in line] for line in english_token_list]
chinese_reflect_data = [[chinese2id[ch] for ch in line] for line in chinese_token_list]

print(f"char(english_data): {english_data[0]}")
print(f"index(english_reflect_data): {english_reflect_data[0]}")
print(f'char(chinese_data): {chinese_data[0]}')
print(f'index(chinese_reflect_data): {chinese_reflect_data[0]}')


def padding_batch(batch):
    '''
    input->list of dict
        [{'source_sample': [1, 2, 3], 'target_sample': [1, 2, 3]}, {'source_sample': [1, 2, 2, 3], 'target_sample': [1, 2, 2, 3]}]
    output->dict of tensor
        {
            "source_sample": [[1, 2, 3, 0], [1, 2, 2, 3]].T
            "target_sample": [[1, 2, 3, 0], [1, 2, 2, 3]].T
        }
    :param batch:
    :return:
    '''

    source_len = [b['source_len'] for b in batch]
    target_len = [b['target_len'] for b in batch]

    source_max_len = max(source_len)
    target_max_len = max(target_len)

    for d in batch:
        d['source_sample'].extend([english2id['<pad>']] * (source_max_len - d['source_len']))
        d['target_sample'].extend([chinese2id['<pad>']] * (target_max_len - d['target_len']))

    source_sample = torch.tensor([pair['source_sample'] for pair in batch], dtype=torch.long, device=device)
    target_sample = torch.tensor([pair['target_sample'] for pair in batch], dtype=torch.long, device=device)

    batch = {'source_sample': source_sample.T, 'source_len': source_len, 'target_sample': target_sample.T,
             'target_len': target_len}

    return batch


# 创建Dataset
class TranslationDataset(Dataset):
    def __init__(self, source_data, target_data):
        self.source_data = source_data
        self.target_data = target_data

        assert len(source_data) == len(target_data), 'numbers of source data and target data must be equal'

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        source_sample = self.source_data[index]
        source_len = len(self.source_data[index])

        target_sample = self.target_data[index]
        target_len = len(self.target_data[index])

        return {'source_sample': source_sample, 'source_len': source_len, 'target_sample': target_sample,
                'target_len': target_len}


'''
Attention机制
'''


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout=0.5, bidirectional=True):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, embedding_dim)

        '''
        GRU:
            Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

            args:
                input_size: The number of expected features in the input `x`
                hidden_size: The number of features in the hidden state `h`
                num_layers: Number of recurrent layers.
                bias: If `False`, then the layer does not use bias weights
                bidirectional: If `True`, becomes a bidirectional GRU.
        '''

        '''
        Inputs: input, h_0
            input: [seq_length, batch, input_size]
            h_0: [num_layers * num_directions, batch_size, hidden_size]

        Outputs: output, h_n
            output: [seq_length, batch_size, num_directions * hidden_size]
            h_n: [num_layers * num_directions, batch_size, hidden_size]
        '''
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_sequence, input_length, h_0):
        '''
        :param input_sequences: [seq_length, batch_size] 一次输入batch个sentence，每个sentence的长度是seq_length
        :param input_length: 一个batch中source sentence的长度列表
        :param h_0:
        :return:
        '''

        # embedding: [seq_length, batch_size, embedding_dim]
        embedding = self.embedding(input_sequence)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedding, input_length, enforce_sorted=False)

        # output: [seq_length, batch_size, num_directions * hidden_size]
        # h_n: [num_layers * num_directions, batch_size, hidden_size]
        output, h_n = self.gru(packed, h_0)

        # output: [seq_length, batch_size, num_directions * hidden_size]
        # output_length=[batch]
        output, output_length = torch.nn.utils.rnn.pad_packed_sequence(output)

        return output, h_n


class Attention(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()

        self.method = method

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    # hidden: [1, batch_size, num_directions*hidden_dim]
    # encoder_output: [seq_length, batch_size, hidden_dim*num_directions]
    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)  # [seq_length, batch_size]

    # hidden: [1, batch_size, num_directions*hidden_dim]
    # encoder_output: [seq_length, batch_size, hidden_dim*num_directions]
    def general_score(self, hidden, encoder_output):
        energy = self.attention(encoder_output)  # [seq_length, batch_size, hidden_dim*num_directions]
        return torch.sum(hidden * energy, dim=2)  # [seq_length, batch_size]

    # hidden: [1, batch_size, num_directions*hidden_dim]
    # encoder_output: [seq_length, batch_size, hidden_dim*num_directions]
    def concat_score(self, hidden, encoder_output):
        interval = torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)

        # energy=[seq_length, batch_size, hidden_dim]
        energy = self.attention(interval).tanh()

        return torch.sum(self.v * energy, dim=2)  # [seq_length, batch_size]

    def forward(self, hidden, encoder_output):
        '''
        :param hidden: [1, batch_size, num_directions*hidden_size]
        :param encoder_output: [seq_length, batch_size, num_directions*hidden_size]
        :return:
        '''
        if self.method == 'general':
            # [seq_length, batch_size]
            attention_energy = self.general_score(hidden, encoder_output)

        elif self.method == 'concat':
            # [seq_length, batch_size]
            attention_energy = self.concat_score(hidden, encoder_output)

        elif self.method == 'dot':
            # [seq_length, batch_size]
            attention_energy = self.dot_score(hidden, encoder_output)

        attention_energy = attention_energy.t()

        return F.softmax(attention_energy, dim=1).unsqueeze(1)  # [batch_size, 1, seq_length]


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers=1, dropout=0.5, bidirectional=True,
                 attention_method='general'):
        super(AttentionDecoder, self).__init__()

        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.embedding_drop = nn.Dropout(dropout)

        '''
                GRU:
                    Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

                    args:
                        input_size: The number of expected features in the input `x`
                        hidden_size: The number of features in the hidden state `h`
                        num_layers: Number of recurrent layers.
                        bias: If `False`, then the layer does not use bias weights
                        bidirectional: If `True`, becomes a bidirectional GRU.
                '''

        '''
        Inputs: input, h_0
            input: [seq_length, batch, input_size]
            h_0: [num_layers * num_directions, batch_size, hidden_size]

        Outputs: output, h_n
            output: [seq_length, batch_size, num_directions * hidden_size]
            h_n: [num_layers * num_directions, batch_size, hidden_size]
        '''

        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional)

        if bidirectional:
            self.concat = nn.Linear(hidden_dim * 2 * 2, hidden_dim * 2)
            self.out = nn.Linear(hidden_dim * 2, output_dim)
            self.attention = Attention(attention_method, hidden_dim * 2)
        else:
            self.concat = nn.Linear(hidden_dim * 2, hidden_dim)
            self.out = nn.Linear(hidden_dim, hidden_dim)
            self.attention = Attention(attention_method, hidden_dim)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, token_inputs, last_hidden, encoder_output):
        '''
        :param token_inputs: 上一个预测作为下一个输入的目标单词
        :param last_hidden:
        :param encoder_output:
        :return:
        '''
        batch_size = token_inputs.size(0)

        embedded = self.embedding(token_inputs)
        embedded = self.embedding_drop(embedded)

        embedded = embedded.view(1, batch_size, -1)  # [1, batch_size, hidden_dim]

        # output: [1, batch_size, num_directions*hidden_dim]
        # hidden: [num_layers*num_directions, batch_size, hidden_dim]
        output, hidden = self.gru(embedded, last_hidden)

        # encoder_output: [seq_length, batch_size, hidden_dim*num_directions]
        # attention_weights: [batch_size, 1, seq_length]
        attention_weights = self.attention(output, encoder_output)

        # 目标单词对应的不同中间状态向量C
        # context: [batch_size, 1, hidden_dim*num_directions]
        context = attention_weights.bmm(encoder_output.transpose(0, 1))

        output = output.squeeze(0)  # [batch_size, num_directions*hidden_dim]
        context = context.squeeze(1)  # [batch_size, num_directions*hidden_dim]
        concat_input = torch.cat((output, context), 1)  # [batch_size, num_directions*hidden_dim*2]
        concat_output = torch.tanh(self.concat(concat_input))  # [batch_size, num_directions*hidden_dim]

        output = self.out(concat_output)  # [batch_size, output_dim]
        output = self.softmax(output)

        return output, hidden, attention_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, predict=False, basic_dict=None, max_len=100):
        super(Seq2Seq, self).__init__()

        self.device = device

        self.encoder = encoder
        self.decoder = decoder

        self.predict = predict  # 训练阶段还是预测阶段
        self.basic_dict = basic_dict  # decoder阶段的字典，存放token到id的映射
        self.max_len = max_len  # 翻译时最大输出长度

        assert encoder.hidden_dim == decoder.hidden_dim, 'Hidden dimensions of encoder and decoder must be equal!'
        assert encoder.n_layers == decoder.n_layers, 'Encoder and decoder must have equal number of layers!'
        assert encoder.gru.bidirectional == decoder.gru.bidirectional, 'Decoder and encoder must had same value of bidirectional attribute!'

    def forward(self, input_batches, input_length, target_batches=None, target_length=None, teacher_forcing_ratio=0.5):
        '''
        :param input_batches: [seq_length, batch_size] 一个batch中的source sentence，每一个sentence中的单词以对应的id表示
        :param input_length: 一个batch中source sentence的长度列表
        :param target_batches: [seq_length, batch_size] 一个batch中的target sentence，每一个sentence中的单词以相应的id表示
        :param target_length: 一个batch中target sentence的长度列表
        :param teacher_forcing_ratio:
        :return:
        '''

        batch_size = input_batches.size(1)

        BOS_token = self.basic_dict['<bos>']
        EOS_token = self.basic_dict['<eos>']
        PAD_token = self.basic_dict['<pad>']

        # 初始化
        encoder_n_layers = self.encoder.n_layers
        encoder_num_directions = 2 if self.encoder.gru.bidirectional else 1

        '''
        Inputs: input, h_0
            input: [seq_length, batch, input_size]
            h_0: [num_layers * num_directions, batch_size, hidden_size]
        '''
        encoder_hidden = torch.zeros(encoder_num_directions * encoder_n_layers, batch_size, self.encoder.hidden_dim,
                                     device=self.device)

        '''
        Outputs: output, h_n
            output: [seq_length, batch_size, num_directions * hidden_size]
            h_n: [num_layers * num_directions, batch_size, hidden_size]
        '''
        encoder_output, encoder_hidden = self.encoder(input_batches, input_length, encoder_hidden)

        # 初始化
        decoder_input = torch.tensor([BOS_token] * batch_size, dtype=torch.long, device=self.device)
        decoder_hidden = encoder_hidden

        if self.predict:
            # 一次只输入一句话
            assert batch_size == 1, "batch_size of predict phase must be 1!"
            output_tokens = []

            while True:
                decoder_output, decoder_hidden, decoder_attention_weights = self.decoder(decoder_input, decoder_hidden,
                                                                                         encoder_output)

                '''
                    top_value: 经过log_softmax计算之后，每一个batch中最大的value
                    top_index: 每一个batch中最大的value对应的index
                '''
                # top_value: [1, 1]
                # top_index: [1, 1]
                top_value, top_index = decoder_output.topk(k=1)

                decoder_input = top_index.squeeze(1).detach()  # 上一个预测作为下一个输入
                output_token = top_index.squeeze().detach().item()

                if output_token == EOS_token or len(output_tokens) == self.max_len:
                    break
                output_tokens.append(output_token)

            return output_tokens

        else:
            max_target_length = max(target_length)
            all_decoder_output = torch.zeros((max_target_length, batch_size, self.decoder.output_dim),
                                             device=self.device)

            for t in range(max_target_length):
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:

                    # decoder_output: [batch_size, output_dim]
                    # decoder_hidden: [n_layers*num_directions, batch_size, hidden_dim]
                    decoder_output, decoder_hidden, decoder_attention_weights = self.decoder(decoder_input,
                                                                                             decoder_hidden,
                                                                                             encoder_output)
                    all_decoder_output[t] = decoder_output
                    decoder_input = target_batches[t]  # 下一个输入来自训练数据

                else:

                    # decoder_output: [batch_size, output_dim]
                    # decoder_hidden: [n_layers*num_directions, batch_size, hidden_dim]
                    decoder_output, decoder_hidden, decoder_attention_weights = self.decoder(decoder_input,
                                                                                             decoder_hidden,
                                                                                             encoder_output)
                    '''
                        top_value: 经过log_softmax计算之后，每一个batch中最大的value
                        top_index: 每一个batch中最大的value对应的index
                    '''
                    # top_value: [batch, 1]
                    # top_index: [batch, 1]
                    top_value, top_index = decoder_output.topk(k=1)

                    all_decoder_output[t] = decoder_output
                    decoder_input = top_index.squeeze(1).detach() # 下一个输入来自上一个输出
            '''
            nn.NLLLoss():
                args:
                    Specifies a target value that is ignored and does not contribute to the input gradient.
            '''
            loss_fn = nn.NLLLoss(ignore_index=PAD_token)
            loss = loss_fn(
                all_decoder_output.reshape(-1, self.decoder.output_dim),  # [batch_size*seq_length, output_dim]
                target_batches.reshape(-1)  # [batch_size*seq_length]
            )
            return loss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_minute = int(elapsed_time / 60)
    elapsed_second = int(elapsed_time - (elapsed_minute * 60))
    return elapsed_minute, elapsed_second


def train(module, dataloader, optimizer, clip=1, teacher_forcing_ratio=0.5):
    module.predict = False
    module.train()

    print_loss_total = 0  # 每次打印都重置

    for i, batch in enumerate(dataloader):
        # source_samples: [seq_length, batch_size]
        # target_samples: [seq_length, batch_size]
        source_samples = batch['source_sample']
        target_samples = batch['target_sample']

        # source_len: 一个batch中source sentence的长度列表
        source_len = batch['source_len']

        # target_len: 一个batch中target sentence的长度列表
        target_len = batch['target_len']

        optimizer.zero_grad()

        loss = module(source_samples, source_len, target_samples, target_len, teacher_forcing_ratio=1)
        loss.backward()

        print_loss_total += loss.item()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(module.parameters(), clip)

        optimizer.step()

        if i % 100 == 0:
            print(f'loss: {loss.item():.4f}')

    return print_loss_total


def evaluate(module, dataloader):
    module.predict = False
    module.eval()

    print_loss_total = 0  # 每次打印都重置

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # source_samples: [seq_length, batch_size]
            # target_samples: [seq_length, batch_size]
            source_samples = batch['source_sample']
            target_samples = batch['target_sample']

            # source_len: 一个batch中source sentence的长度列表
            source_len = batch['source_len']

            # target_len: 一个batch中target sentence的长度列表
            target_len = batch['target_len']

            loss = module(source_samples, source_len, target_samples, target_len, teacher_forcing_ratio=0)

            print_loss_total += loss.item()

    return print_loss_total


def translate(module, sample, index2token=None):
    module.predict = True
    module.eval()

    # source_sample: [seq_length, 1]
    source_sample = sample['source_sample']

    source_len = sample['source_len']

    output_tokens = module(source_sample, source_len)
    output_tokens = [index2token[t] for t in output_tokens]

    return ''.join(output_tokens)


# 超参数
BATCH_SIZE = 32
INPUT_DIM = len(english2id)
OUTPUT_DIM = len(chinese2id)
ENCODER_EMBEDDING_DIM = 256
DECODER_EMBEDDING_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
ENCODER_DROPOUT = 0.5
DECODER_DROPOUT = 0.5
LEARNING_RATE = 1e-4
N_EPOCHS = 200
CLIP = 1
BIDIRECTIONAL = True

encoder = Encoder(INPUT_DIM, ENCODER_EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, ENCODER_DROPOUT, BIDIRECTIONAL)
decoder = AttentionDecoder(OUTPUT_DIM, DECODER_EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DECODER_DROPOUT, BIDIRECTIONAL)
module = Seq2Seq(encoder, decoder, device, basic_dict=basic_dictionary).to(device)

optimizer = optim.Adam(module.parameters(), lr=LEARNING_RATE)

train_dataset = TranslationDataset(source_data=english_reflect_data, target_data=chinese_reflect_data)
print(f'train_dataset[0]: {train_dataset[0]}')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=padding_batch, shuffle=True)

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    print(f'{"-" * 20}第{epoch + 1}次训练{"-" * 20}')
    start_time = time.time()
    train_loss = train(module, train_dataloader, optimizer, CLIP)
    valid_loss = evaluate(module, train_dataloader)
    end_time = time.time()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(module.state_dict(), 'english2chinese-model.pt')

    if epoch % 2 == 0:
        epoch_minute, epoch_second = epoch_time(start_time, end_time)
        print(f'epoch: {epoch} | time: {epoch_minute} min {epoch_second} sec')
        print(f'train loss: {train_loss:.3f} | valid loss: {valid_loss:.3f}')