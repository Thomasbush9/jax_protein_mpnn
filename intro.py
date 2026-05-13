import torch 
import tqdm
import optax
from torch.nn.modules import loss
from torch.utils.data import DataLoader, Dataset, default_collate
import jax.numpy as jnp 
from jax.tree_util import tree_map 

# dataset def 
class TitanicDataset(Dataset):
    def __init__(self, samples, labels) -> None:
        self.df = samples 
        self.labels = labels


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.df.iloc[idx].values, 
                         dtype=torch.float32)
        y = torch.tensor(self.labels.iloc[idx].values,
                         dtype=torch.float32)
        return x, y
    
def numpy_collate(batch):
    return tree_map(jnp.asarray, default_collate(batch))

import torch.nn as nn
# torch model def: 
class NeuralNet(nn.Module):
    def __init__(self, num_hidden_1, num_hidden_2)->None:
        super().__init__()
        self.linear1 = nn.Linear(8, num_hidden_1)
        self.dropout = nn.Dropout(.8)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.linear3 = nn.Linear(num_hidden_2, 1, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.linear3(x)
        return out
# model using flax 
#
from flax import nnx
class NNX(nnx.Module):
    def __init__(self, num_hidden_1, num_hidden_2, rngs:nnx.Rngs) -> None:
        self.linear1 = nnx.Linear(8, num_hidden_1, rngs=rngs)
        self.dropout = nnx.Dropout(0.01, rngs=rngs)
        self.relu = nnx.leaky_relu
        self.linear2 = nnx.Linear(num_hidden_1, num_hidden_2, rngs=rngs)
        self.linear3 = nnx.Linear(num_hidden_2, 1, use_bias=False, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.linear3(x)
        return out 

# we can init the models like this: 
torch_model = NeuralNet(
    num_hidden_1=32,
    num_hidden_2=16
)


flax_model = NNX(
    num_hidden_1=32, 
    num_hidden_2=16,
    rngs=nnx.Rngs(0)
)


#process a batch of data
flax_model(sample_data)

# optimization + backpropagation:
optmizer = nnx.Optimizer(model, optax.adam(learning_rate=0.01))
# define loss function: 

def loss_fn(model):
    logits = model(batch)
    loss = optax.sigmoid_binary_cross_entropy(logits.squeeze(), labels).mean()
    return loss
grad_fn = nnx.value_and_grad(loss_fn)
loss, grads = grad_fn(model)
# optmizer step 
optmizer.update(grads)

# training loop: 
def train(
    model, 
    train_dataloader, 
    eval_dataloader, 
    num_epochs, 


):
    optmizer = nnx.Optimizer(
        model, 
        optax.adam(learning_rate=.)
    )
    for epoch in (pbar := tqdm(range(num_epochs))):
        pbar.set_description(f"Epoch {epoch}")
        model.train()
        for batch in train_dataloader:
            train_step(model, optimizer, batch)

        pbar.set_potfix(train_accuracy=eval(model, train_dataloader), eval_accuracy=eval(model, eval_dataloader))

@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch[0])
        loss = optax.sigmoid_binary_cross_entropy(logits.squeeze(), batch[1]).mean()
        return loss 
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    optimizer.update(grads)


def eval(model, eval_dataloader):
    model.eval()
    total=0
    num_correct=0
    for batch in eval_dataloader:
        res = eval_step(model, batch)
        total += res.shape[0]
        num_correct += jnp.sum(res)
    return num_correct / total 

@nnx.jit
def eval_step(model , batch): 
    logits = model(batch[0])
    logits = logits.squeeze()
    preds = jnp.round(nnx.sigmoid(logits))
    return preds == batch[1]
               
