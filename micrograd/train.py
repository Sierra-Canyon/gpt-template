# micrograd/train.py  (T11, T12)
# Run from the repo root:  uv run python micrograd/train.py before     (T11: one forward pass of an untrained net, and its loss)
#                     or:  uv run python micrograd/train.py            (T12: train it, loss every 10 steps, the curve, the misses)
#                     or:  uv run python micrograd/train.py noreset    (T12 extension: the same training without zeroing the grads)
# The network is YOUR MLP from micrograd/nn.py, 2 inputs, two hidden layers of 4, one output, on the 40 corpus slices
# from micrograd/slices.py. The loss is the chapter's: the sum over slices of (prediction - label) squared.
# Nothing to edit in this file.
import sys
import random
sys.path.insert(0, "micrograd")
import matplotlib
matplotlib.use("Agg")                              # draw to a file, not a window
import matplotlib.pyplot as plt
from nn import MLP
from slices import load

STEPS = 100
LR = 0.01

xs, ys, where = load()
random.seed(0)                                     # everyone's net starts from the same random weights
net = MLP(2, [4, 4, 1])

# In: the net.  Out: its predictions for every slice and the summed squared loss, as Values.
def forward(net):
    preds = [net(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
    return preds, loss

# In: the predictions.  Out: how many slices land on the right side of 0.
def correct(preds):
    return sum((p.data > 0) == (y > 0) for p, y in zip(preds, ys))

mode = sys.argv[1] if len(sys.argv) > 1 else "train"
if mode == "noreset":
    STEPS = 200                                    # twice as long, so what the stale gradients do has time to show
preds, loss = forward(net)
print(f"parameters: {len(net.parameters())}")
print(f"before training: loss {loss.data:.4f}   correct {correct(preds)}/{len(xs)}")
if mode == "before":
    print("first five predictions:", [round(p.data, 3) for p in preds[:5]], "labels", ys[:5])
    sys.exit()

losses = []
blew_up = False                                    # set when a number grows past what the engine's tanh can hold
for step in range(STEPS):
    try:
        preds, loss = forward(net)                 # forward: predictions and the loss
    except OverflowError:                          # math.exp inside tanh was handed a number too large: the net has exploded
        print(f"step {step:3d}   OverflowError: a neuron's sum is too large for tanh, so training stops here")
        blew_up = True
        break
    if mode != "noreset":
        for p in net.parameters():
            p.grad = 0.0                           # the gradients ADD UP (+=), so they have to be zeroed before each backward
    loss.backward()                                # backward: every parameter gets its gradient
    for p in net.parameters():
        p.data -= LR * p.grad                      # the update: downhill, against the gradient
    losses.append(loss.data)
    if step % 10 == 0 or step == STEPS - 1:
        print(f"step {step:3d}   loss {loss.data:8.4f}   correct {correct(preds)}/{len(xs)}")

if not blew_up:
    preds, loss = forward(net)                     # one more forward pass, so the numbers below are after the last update
print(f"after training: loss {loss.data:.4f}   correct {correct(preds)}/{len(xs)}")
print("misses (chapter, start, x1, x2, label, prediction):")
for (c, start), x, y, p in zip(where, xs, ys, preds):
    if (p.data > 0) != (y > 0):
        print(f"   chapter {c}  start {start:5d}   x1 {x[0]:6.2f}  x2 {x[1]:6.2f}   label {y:+.0f}   predicted {p.data:+.3f}")

name = "micrograd/loss_noreset.png" if mode == "noreset" else "micrograd/loss.png"
plt.figure(figsize=(6, 4))
plt.plot(range(len(losses)), losses)               # len(losses) is STEPS, or fewer if training stopped early
plt.xlabel("step"); plt.ylabel("loss (sum of squared error)"); plt.title("MLP(2, [4, 4, 1]) on 40 corpus slices")
plt.tight_layout(); plt.savefig(name, dpi=110)
print("drawn:", name)
