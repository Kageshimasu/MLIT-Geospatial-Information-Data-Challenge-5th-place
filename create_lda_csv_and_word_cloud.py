import pandas as pd
import numpy as np
import torch
import pyro
from pyro.infer import SVI, TraceMeanField_ELBO, Predictive
from tqdm import trange
import math
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
import gensim
from gensim.models import CoherenceModel

class Encoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

    # 生成モデル p(docs|θ)
    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # θの事前分布
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # 生成分布 p(docs|θ)
            count_param = self.decoder(theta)
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs',
                dist.Multinomial(total_count, count_param),
                obs=docs
            )

    # 推論モデル q(θ|docs)
    def guide(self, docs):
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))

    def beta(self):
        return self.decoder.beta.weight.cpu().detach().T

def plot_word_cloud(b, ax, vocab, n):
    sorted_, indices = torch.sort(b, descending=True)
    indices = indices[:100].numpy()
    words = vocab.iloc[indices]['word'].values.tolist()
    sizes = (sorted_[:100] * 1000).int().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(
        background_color="white",
        width=800,
        height=500,
        font_path='./font.ttf'
    )
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")


tokyo_data = pd.read_csv('./lda_feature_5.csv')
non_binary_columns = ['target_ym', 'money_room', 'lat', 'lon', 'unit_area', 'addr1_1', 'addr1_2', 'walk_distance1']
binary_columns = tokyo_data.columns.difference(non_binary_columns)

test_ratio = 0.2
random_seed = 42

tokyo_data_sampled = tokyo_data
train_docs = tokyo_data_sampled[binary_columns].values  # NumPy配列
train_docs = torch.tensor(train_docs, dtype=torch.float32)
vocab = pd.DataFrame({'word': binary_columns, 'index': range(len(binary_columns))})

print('Dictionary size: %d' % len(vocab))
print('Train Corpus size: {}'.format(train_docs.shape))

seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32
learning_rate = 1e-3
num_epochs = 20

# トピック数のリスト
topic_nums = [5, 7, 10, 13, 15, 17, 20]
perplexities = []

for num_topics in topic_nums:
    print(f'\nTraining model with {num_topics} topics...')

    pyro.clear_param_store()
    
    prodLDA = ProdLDA(
        vocab_size=train_docs.shape[1],
        num_topics=num_topics,
        hidden=100,
        dropout=0.2
    )
    prodLDA.to(device)
    
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(math.ceil(train_docs.shape[0] / batch_size))
    
    train_docs = train_docs.to(device)

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_docs = train_docs[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_docs)
            running_loss += loss
        epoch_loss = running_loss / len(train_docs)
        bar.set_postfix(epoch_loss='{:.2e}'.format(epoch_loss))

    # トピックの割合を取得し、物件ごとのデータに結合して保存
    # Predictiveを使用して潜在変数を推論
    predictive = Predictive(prodLDA.model, guide=prodLDA.guide, num_samples=1)
    svi_samples = predictive(train_docs)

    # logthetaを取得し、thetaを計算
    logtheta = svi_samples["logtheta"].squeeze(0) 
    theta = F.softmax(logtheta, dim=-1).detach().cpu().numpy()

    theta_df = pd.DataFrame(theta, columns=[f'topic_{i}' for i in range(num_topics)])
    tokyo_data_with_topics = pd.concat([tokyo_data, theta_df], axis=1)
    tokyo_data_with_topics.to_csv(f"./tokyo_data_with_topics_5_topic{num_topics}.csv", index=False)

    beta = prodLDA.beta()
    fig, axs = plt.subplots(7, 3, figsize=(14, 24))
    axs = axs.flatten()
    for n in range(num_topics):
        plot_word_cloud(beta[n], axs[n], vocab, n)
    for ax in axs[num_topics:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"./word_cloud_5_topic{num_topics}.png")
