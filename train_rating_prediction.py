# %%
import torch
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torchsummary import summary

# %%
df = pd.read_csv("data/RatingPrediction/rating_reviewer_pairs.csv")

# %%
df.dropna(inplace=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
print(df_train.shape, df_test.shape)

scaler = StandardScaler()
df_train = scaler.fit_transform(df_train)
df_test = scaler.transform(df_test)

# df_train[:, 9:317] = 0  # No text
# df_test[:, 9:317] = 0  # No text
# df_train[:, 330:342] = 0  # Month of review
# df_test[:, 330:342] = 0  # Month of review

# %% [markdown]
# Dataframe shape: 342 columns
# Overview:
# - df.columns[1:9]:
#     - label_actual_rating,
#     - user_mean_until_now,
#     - user_style_mean_until_now,
#     - user_num_ratings_until_now,
#     - is_exp,
#     - user_brewery_distance,
#     - beer_mean_rating,
#     - interaction_good,
#     - interaction_bad,
# - "good_distr_user"  # 77 vals, df.columns[9:86]
# - "bad_distr_user"   # 77 vals, df.columns[86:163]
# - "good_distr_beer"  # 77 vals, df.columns[163:240]
# - "bad_distr_beer"  # 77 vals, df.columns[240:317]
# - "one_hot_cat"   # 13 vals, df.columns[317:330]
# - one_hot_month  # 12 vals, df.columns[330:342]
#
# => 342 columns
#
# Will be converted to:
# y_label: label_actual_rating
#
# **Inputs:**
# 1. "Classical Biases": Is user exp?, Distance, Interaction good/bad, user mean rating,... (8 cols)
# 2. 4x Linear Embedding(77, 5) for each of the vocab distributions (4 x 5 cols = 20)
# 3. Beer Category (13 cats, 13 cols)
# 4. Month of review (12 months, 12 cols)
#
# => 8 + 20 + 13 + 12 = 53 inputs <br>
# => 1 Regression Output or Good/Bad Output
#
# *Not sure about colinearity of categorical features, could probably omit one...*


# %%
class RatingModel(torch.nn.Module):
    def __init__(self):
        super(RatingModel, self).__init__()
        self.embd_good_word_user = torch.nn.Linear(77, 15)
        self.embd_bad_word_user = torch.nn.Linear(77, 15)
        self.embd_good_word_beer = torch.nn.Linear(77, 15)
        self.embd_bad_word_beer = torch.nn.Linear(77, 15)

        self.first_layer = torch.nn.Linear(92, 52)
        self.second_layer = torch.nn.Linear(52, 35)
        self.third_layer = torch.nn.Linear(35, 35)
        self.fourth_layer = torch.nn.Linear(35, 35)
        self.fifth_layer = torch.nn.Linear(35, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        classical_biases = x[:, 1:9]
        user_good_words = x[:, 9:86]
        user_bad_words = x[:, 86:163]
        beer_good_words = x[:, 163:240]
        beer_bad_words = x[:, 240:317]
        beer_cat = x[:, 317:330]
        month = x[:, 330:342]

        user_good_words = self.relu(self.embd_good_word_user(user_good_words))
        user_bad_words = self.relu(self.embd_bad_word_user(user_bad_words))
        beer_good_words = self.relu(self.embd_good_word_beer(beer_good_words))
        beer_bad_words = self.relu(self.embd_bad_word_beer(beer_bad_words))

        x = torch.cat(
            [
                classical_biases,
                user_good_words,
                user_bad_words,
                beer_good_words,
                beer_good_words,
                beer_cat,
                month,
            ],
            dim=1,
        )
        x = self.dropout(x)
        x = self.relu(self.first_layer(x))
        x = self.relu(self.second_layer(x))
        x = self.relu(self.third_layer(x))
        x = self.relu(self.fourth_layer(x))
        x = self.sigmoid(self.fifth_layer(x))

        return x


# %%
train_ds = torch.utils.data.TensorDataset(
    torch.tensor(df_train[:, 1:], dtype=torch.float32),
    torch.tensor(df_train[:, 0] > df_train[:, 1], dtype=torch.float32),
)
test_ds = torch.utils.data.TensorDataset(
    torch.tensor(df_test[:, 1:], dtype=torch.float32),
    torch.tensor(df_test[:, 0] > df_test[:, 1], dtype=torch.float32),
)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=256, shuffle=True, num_workers=8, persistent_workers=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=256, shuffle=False, num_workers=8, persistent_workers=True
)

# %%
model = RatingModel()
crit = torch.nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=5e-4)

# %%

# print(summary(model, (1, 342)))
model.cuda()
losses_train = []
losses_test = []
accs_train = []
accs_test = []

best_acc_test = 0
for epoch in tqdm(range(100)):
    print(f"Epoch {epoch}")
    model.train()
    loss_train = []
    acc_train = []

    for x, y in tqdm(train_loader):
        optim.zero_grad()
        # model expects: classical_biases, user_good_words, user_bad_words, beer_good_words, beer_bad_words, beer_cat, month
        y_pred = model(x.cuda())
        y.cuda()
        loss = crit(y_pred, y.unsqueeze(1).cuda())
        loss.backward()
        optim.step()
        loss_train.append(loss.item())
        acc_train.append(
            accuracy_score(
                y.cpu().detach().numpy(), y_pred.cpu().detach().numpy() > 0.5
            )
        )

    model.eval()
    loss_test = []
    acc_test = []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            y_pred = model(x.cuda())
            loss = crit(y_pred, y.unsqueeze(1).cuda())
            loss_test.append(loss.item())
            acc_test.append(accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy() > 0.5))

    if best_acc_test < np.mean(acc_test):
        best_acc_test = np.mean(acc_test)
        torch.save(model.state_dict(), "best_model.pth")

    # plt.plot(loss_train)
    # plt.plot(loss_test)
    # plt.show()
    losses_train.append(np.mean(loss_train))
    losses_test.append(np.mean(loss_test))
    accs_train.append(np.mean(acc_train))
    accs_test.append(np.mean(acc_test))
    print(
        f"Train loss: {np.mean(loss_train)}, Test loss: {np.mean(loss_test)}, Train acc: {np.mean(acc_train)}, Test acc: {np.mean(acc_test)}"
    )


plt.plot(losses_train, label="train")
plt.plot(losses_test, label="test")
plt.legend()
plt.title("Losses")
plt.savefig("losses_relu_embd_15.png")

plt.figure()
plt.plot(accs_train, label="train")
plt.plot(accs_test, label="test")
plt.legend()
plt.title("Accuracies")
plt.savefig("accs_relu_embd_15.png")
