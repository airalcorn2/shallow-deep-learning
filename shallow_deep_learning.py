import numpy as np
import torch
import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset

# Python things.

# List.
my_list = [0, "one"]  # You can initialize an empty list with [].
# Indexing.
print(my_list[0])
print(my_list[1])
# Appending.
my_list.append(pow)
print(my_list[2](2, 3))  # Equivalent to pow(2, 3).
# List comprehension.
my_list = [(x - 1) / 3 for x in range(10, 20, 3)]
# Equivalent to:
my_list = []
for x in range(10, 20, 3):
    my_list.append((x - 1) / 3)

# Dictionary.
my_dict = {"a": 1, 3: "three"}  # You can initialize an empty dictionary with {}.
# Accessing.
print(my_dict["a"])
print(my_dict[3])
# Inserting.
my_dict["my_list"] = my_list

# NumPy things.
shape = (50, 10)
my_array = np.random.normal(size=shape)
print(my_array.shape)
(row_idx, col_idx) = (20, 5)
print(my_array[row_idx, col_idx])
# Access all rows and a single column.
print(my_array[:, col_idx])
# Access all rows except the first three and the last three and a single column.
print(my_array[3:-3, col_idx])

# Linear algebra.
(m, n, k) = (10, 20, 5)
A = np.random.normal(size=(m, n))
B = np.random.normal(size=(n, k))
# A @ B is equivalent to np.matmul(A, B).
C = A @ B

# Fully connected layer example.

# NumPy.
batch_size = 32
features = 100
X = np.random.normal(size=(batch_size, features))

hidden_nodes = 50
W = np.random.normal(size=(hidden_nodes, features))
b = np.random.normal(size=(hidden_nodes, 1))

out = W @ X.T + b
# Rectifier activation function.
out[out < 0] = 0
print(out.T)

# PyTorch.
fc = nn.Linear(features, hidden_nodes)
# I'm loading the weights so the output will be identical to the NumPy results, but you
# don't usually do this.
with torch.no_grad():
    fc.weight[:, :] = torch.Tensor(W)
    fc.bias[:] = torch.Tensor(b.flatten())

out = fc(torch.Tensor(X))
out = nn.functional.relu(out)
# Same as above.
print(out)

# Embedding example.

# NumPy.
n_batters = n_pitchers = 100
batter_idxs = np.random.randint(n_batters, size=batch_size)
batter_one_hots = np.zeros((batch_size, n_batters))
batter_one_hots[np.arange(batch_size), batter_idxs] = 1
pitcher_idxs = np.random.randint(n_pitchers, size=batch_size)
pitcher_one_hots = np.zeros((batch_size, n_pitchers))
pitcher_one_hots[np.arange(batch_size), pitcher_idxs] = 1

embedding_dim = 9
W_b = np.random.normal(size=(embedding_dim, n_batters))
W_p = np.random.normal(size=(embedding_dim, n_pitchers))

# Equivalent to:
# batter_embeds = W_b[:, batter_idxs]
batter_embeds = W_b @ batter_one_hots.T
# Equivalent to:
# pitcher_embeds = W_p[:, pitcher_idxs]
pitcher_embeds = W_p @ pitcher_one_hots.T

cat_embeds = np.hstack([batter_embeds.T, pitcher_embeds.T])
print(cat_embeds[[0, -1]])

# PyTorch.
batter_embed = nn.Embedding(n_batters, embedding_dim)
pitcher_embed = nn.Embedding(n_pitchers, embedding_dim)

with torch.no_grad():
    batter_embed.weight[:, :] = torch.Tensor(W_b.T)
    pitcher_embed.weight[:, :] = torch.Tensor(W_p.T)

batter_embeds = batter_embed(torch.LongTensor(batter_idxs))
pitcher_embeds = pitcher_embed(torch.LongTensor(pitcher_idxs))
cat_embeds = torch.cat([batter_embeds, pitcher_embeds], dim=1)
print(cat_embeds[[0, -1]])

# Building a model.


class BatterPitcher2Vec(nn.Module):
    def __init__(self, n_batters, n_pitchers, embedding_dim, n_outcomes):
        super().__init__()
        self.batter_embed = nn.Embedding(n_batters, embedding_dim)
        self.pitcher_embed = nn.Embedding(n_pitchers, embedding_dim)
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(2 * embedding_dim, n_outcomes)

    def forward(self, batter_idxs, pitcher_idxs):
        batter_embeds = self.batter_embed(batter_idxs)
        pitcher_embeds = self.pitcher_embed(pitcher_idxs)
        cat_embeds = torch.cat([batter_embeds, pitcher_embeds], dim=1)
        return self.fc(self.sig(cat_embeds))


n_outcomes = 20
model = BatterPitcher2Vec(n_batters, n_pitchers, embedding_dim, n_outcomes)
print(model)
batch_size = 32
test_batters = torch.randint(n_batters, (batch_size,))
test_pitchers = torch.randint(n_pitchers, (batch_size,))
out = model(test_batters, test_pitchers)

# Training a model.


class BatterPitcher2VecDataset(Dataset):
    def __init__(self):
        N = 128
        self.batter_idxs = np.random.randint(n_batters, size=N)
        self.pitcher_idxs = np.random.randint(n_pitchers, size=N)
        self.outcomes = np.random.randint(n_outcomes, size=N)

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        return {
            "batter": torch.LongTensor([self.batter_idxs[idx]]),
            "pitcher": torch.LongTensor([self.pitcher_idxs[idx]]),
            "outcome": torch.LongTensor([self.outcomes[idx]]),
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BatterPitcher2Vec(n_batters, n_pitchers, embedding_dim, n_outcomes).to(device)

criterion = nn.CrossEntropyLoss()
train_params = [params for params in model.parameters()]
learning_rate = 1e-1
optimizer = torch.optim.Adam(train_params, lr=learning_rate)

train_dataset = BatterPitcher2VecDataset()
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = train_loader

# Train model.
epochs = 200
best_val_loss = np.inf
for epoch in range(epochs):
    model.train()
    for train_tensors in train_loader:
        optimizer.zero_grad()
        pred_logits = model(
            train_tensors["batter"].flatten().to(device),
            train_tensors["pitcher"].flatten().to(device),
        )
        loss = criterion(pred_logits, train_tensors["outcome"].flatten().to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    for valid_tensors in valid_loader:
        with torch.no_grad():
            pred_logits = model(
                valid_tensors["batter"].flatten().to(device),
                valid_tensors["pitcher"].flatten().to(device),
            )
            val_loss += criterion(
                pred_logits, valid_tensors["outcome"].flatten().to(device)
            ).item()

    print(val_loss, flush=True)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "batter_pitcher2vec.pth")

model.load_state_dict(torch.load("batter_pitcher2vec.pth"))

# Convolutional layer example.

# NumPy.
rgb_feats = 3
img_size = 4
X = np.random.normal(size=(batch_size, rgb_feats, img_size, img_size))

num_filters = 5
filter_size = 3
W = np.random.normal(size=(num_filters, rgb_feats, filter_size, filter_size))
b = np.random.normal(size=(num_filters,))

# Only process one image.
in_img = X[0]
outs = []
for filter_idx in range(num_filters):
    filter_W = W[filter_idx].flatten()
    filter_b = b[filter_idx]
    filter_outputs = []
    out_size = img_size - filter_size + 1
    for win_row in range(out_size):
        for win_col in range(out_size):
            img_win = in_img[
                :, win_row : win_row + filter_size, win_col : win_col + filter_size
            ]
            filter_outputs.append(filter_W @ img_win.flatten() + filter_b)

    filter_outputs = np.array(filter_outputs).reshape((out_size, out_size))
    outs.append(filter_outputs)

out = np.stack(outs)
out[out < 0] = 0
print(out)

# PyTorch.
conv = nn.Conv2d(rgb_feats, num_filters, filter_size)
with torch.no_grad():
    conv.weight[:, :, :, :] = torch.Tensor(W)
    conv.bias[:] = torch.Tensor(b)

out = conv(torch.Tensor(X))
out = nn.functional.relu(out)
print(out[0])

# Average pooling example.

avg_pool = nn.AdaptiveAvgPool2d((1, 1))
out = avg_pool(out)
print(out.shape)

# Recurrent neural network example.

# NumPy.
seq_len = 10
X = np.random.random(size=(seq_len, batch_size, features))
W_h = np.random.normal(size=(hidden_nodes, features))
U_h = np.random.normal(size=(hidden_nodes, hidden_nodes))
b_h = np.random.normal(size=(hidden_nodes, 1))

# Only process one sequence.
in_seq = X[:, 0]
h = np.zeros((hidden_nodes, 1))
out_hs = []
for step in range(seq_len):
    # None index keeps the dimensions the same as the input.
    h = np.tanh(W_h @ in_seq[None, step].T + U_h @ h + b_h)
    out_hs.append(h)

out = np.hstack(out_hs).T
print(out)

# PyTorch.
rnn = nn.RNN(features, hidden_nodes)
with torch.no_grad():
    rnn.weight_ih_l0[:, :] = torch.Tensor(W_h)
    rnn.weight_hh_l0[:, :] = torch.Tensor(U_h)
    rnn.bias_ih_l0[:] = torch.Tensor(b_h.flatten())
    rnn.bias_hh_l0[:] = 0

(out, last_h) = rnn(torch.Tensor(X))
print(out[:, 0])

# Self-attention example.

# PyTorch.
(h1, h2) = (50, 30)
attn = nn.Sequential(
    nn.Linear(2 * features, h1),
    nn.ReLU(),
    nn.Linear(h1, h2),
    nn.ReLU(),
    nn.Linear(h2, 1),
)

total_time_steps = 10
X = torch.rand(total_time_steps, features)
t = 7
x_t = X[None, t]
pre_t_xs = X[:t]
pre_t_xs_w_x_t = torch.cat([pre_t_xs, x_t.expand(t, -1)], dim=1)
scores = attn(pre_t_xs_w_x_t)
c = pre_t_xs.T @ scores
x_t_with_c = torch.cat([x_t, c.T], dim=1)
print(x_t_with_c)

# Transformer example.

# PyTorch.
# Same input as RNN example above.
X = np.random.random(size=(seq_len, batch_size, features))
nhead = 5
encoder_layers = TransformerEncoderLayer(features, nhead, hidden_nodes)
num_layers = 3
transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
seq_mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
seq_mask = (
    seq_mask.float()
    .masked_fill(seq_mask == 0, float("-inf"))
    .masked_fill(seq_mask == 1, float(0.0))
)
out = transformer_encoder(torch.Tensor(X), seq_mask)
print(out[:, 0])
