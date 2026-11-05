# names/mlp.py  (T17, T18, T19, T20)
# Run from the repo root:  uv run python names/mlp.py
# The makemore MLP from the second video, on YOUR names: three characters of context, each looked up in a small
# embedding table, the three embeddings side by side into a hidden layer of tanh neurons, then logits for the 27
# possible next characters. Run directly, it builds the dataset, shows the first name's contexts, builds the untrained
# network with the video's seed, and runs one forward pass with every shape printed. Imported by train_mlp.py,
# embed_plot.py and demo.py for the dataset and the network. Nothing to edit in this file.
import sys
sys.path.insert(0, "names")
import torch
import torch.nn.functional as F
from counts import V, itos, stoi, words

BLOCK = 3                                          # how many characters of context the model sees
EMB = 2                                            # how many numbers each character becomes
HIDDEN = 100                                       # neurons in the hidden layer
SEED = 2147483647

# In: a list of names.  Out: X, every context of BLOCK characters as integers, and Y, the character that followed each.
def build_dataset(names):
    X, Y = [], []
    for w in names:
        context = [0] * BLOCK                      # a name starts with BLOCK dots of context
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]           # slide the window: drop the oldest character, add this one
    return torch.tensor(X), torch.tensor(Y)

# In: a generator.  Out: the five parameter tensors, random, with requires_grad on.
def make_parameters(g):
    C = torch.randn((V, EMB), generator=g)                  # the embedding table: one row of EMB numbers per character
    W1 = torch.randn((BLOCK * EMB, HIDDEN), generator=g)    # the hidden layer reads BLOCK embeddings side by side
    b1 = torch.randn(HIDDEN, generator=g)
    W2 = torch.randn((HIDDEN, V), generator=g)              # the output layer: one logit per possible next character
    b2 = torch.randn(V, generator=g)
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True
    return parameters

# In: the parameters and a batch of contexts.  Out: the logits, one row per context.
def forward(parameters, X):
    C, W1, b1, W2, b2 = parameters
    emb = C[X]                                     # (n, BLOCK, EMB): each context's characters looked up in the table
    h = torch.tanh(emb.view(-1, BLOCK * EMB) @ W1 + b1)    # (n, HIDDEN): the embeddings side by side, through the hidden layer
    return h @ W2 + b2                             # (n, V): the logits

if __name__ == "__main__":
    X, Y = build_dataset(words)
    print(f"{len(words)} names -> {X.shape[0]} examples   X {tuple(X.shape)}   Y {tuple(Y.shape)}")
    w = words[0]
    print(f"the contexts of {w!r}:")
    for x, y in zip(X[:len(w) + 1], Y[:len(w) + 1]):
        print("   ", "".join(itos[int(i)] for i in x), "-->", itos[int(y)])

    g = torch.Generator().manual_seed(SEED)
    parameters = make_parameters(g)
    C, W1, b1, W2, b2 = parameters
    print(f"C {tuple(C.shape)}   W1 {tuple(W1.shape)}   b1 {tuple(b1.shape)}   W2 {tuple(W2.shape)}   b2 {tuple(b2.shape)}")
    print(f"parameters: {sum(p.nelement() for p in parameters)}")

    # one forward pass on the first name's contexts, every shape shown
    Xw, Yw = X[:len(w) + 1], Y[:len(w) + 1]
    emb = C[Xw]
    hpre = emb.view(-1, BLOCK * EMB) @ W1 + b1
    h = torch.tanh(hpre)
    logits = h @ W2 + b2
    probs = F.softmax(logits, dim=1)
    print(f"emb {tuple(emb.shape)} -> view {tuple(emb.view(-1, BLOCK * EMB).shape)} -> h {tuple(h.shape)} -> logits {tuple(logits.shape)} -> probs {tuple(probs.shape)}")
    print(f"row of C for {w[0]!r}: {[round(float(v), 3) for v in C[stoi[w[0]]].detach()]}")
    loss = F.cross_entropy(logits, Yw)
    print(f"untrained loss on {w!r}'s {len(w) + 1} examples: {loss.item():.4f}   (a uniform guess scores {torch.log(torch.tensor(float(V))).item():.4f})")
    print(f"probability the untrained net gave the real next character, per context: {[round(float(p), 4) for p in probs[torch.arange(len(Yw)), Yw].detach()]}")
