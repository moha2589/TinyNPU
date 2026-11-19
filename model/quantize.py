import torch
import numpy as np
from net import NeuralNetwork, get_dataloaders

model = NeuralNetwork()
state = torch.load("model/model_float.pth")
model.load_state_dict(state)
model.eval()

w1 = model.linear_relu_stack[0].weight.detach()
w2 = model.linear_relu_stack[2].weight.detach()

# Scale weights

# First, find the maximum weight and divide that by the largest 8bit int to define that as the scale (No weight multiplied by our scale will be larger than 127)
scale1 = w1.abs().max() / 127
w1_int8 = torch.round(w1 / scale1).clamp(-128, 127).to(torch.int8)

scale2 = w2.abs().max() / 127
w2_int8 = torch.round(w2 / scale2).clamp(-128, 127).to(torch.int8)

train_dataloader, test_dataloader = get_dataloaders(batch_size=64)

scale_x = 1 / 127


w1_int32 = w1_int8.to(torch.int32).T


# Now we need to scale the hidden layers inputs, remember, 784 -> 32 -> 10
first_half = model.linear_relu_stack[:2]

largest = 0
with torch.no_grad():
    for X_2, z in train_dataloader:
        hidden = first_half(model.flatten(X_2))
        largest = max(hidden.max().item(), largest)

scale_hidden = largest / 127

# Must change from ugly float to integer  * base 2 ^ shift

M = (scale1 * scale_x) / scale_hidden

shift = 16

m = round(M.item() * 2**shift)


correct = 0
# Weights and scales do not need to be recalculated for each image. The inputs themselves do, however
for X, y in test_dataloader:
    # Then, we must scale the first input. Here we know that the brightest a pixel can be is 1 so we again divide that by 127
    X = X.flatten(1)
    X_int8 = torch.round(X / scale_x).clamp(0, 127).to(torch.int8)
    X_int32 = X_int8.to(torch.int32)
    # After transposing our weights to multiply the weight and input matrix, we set that as our "accumulator", which we will add to in later steps.
    accumulator = X_int32 @ w1_int32
    hidden_int8 = (
        ((accumulator * m + (1 << (shift - 1))) >> shift).clamp(0, 127).to(torch.int8)
    )
    predictions = (hidden_int8.to(torch.int32) @ w2_int8.to(torch.int32).T).argmax(1)

    # For each batch (64 images), add all the correct images and add it as a number (56 Trues adds 56 to correct)
    correct += (predictions == y).sum().item()

print(f"accuracy: {100 * correct / len(test_dataloader.dataset):.2f}%")

np.savez("golden/w_int8.npz", w1=w1_int8.numpy(), w2=w2_int8.numpy(), m=m, shift=shift)
