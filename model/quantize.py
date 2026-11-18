import torch
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

# What does this line do again
X, y = next(iter(test_dataloader))


# Then, we must scale the first input. Here we know that the brightest a pixel can be is 1 so we again divide that by 127
X = X.flatten(1)

scale_x = 1 / 127

X_int8 = torch.round(X / scale_x).clamp(0, 127).to(torch.int8)

w1_int32 = w1_int8.to(torch.int32).T
X_int32 = X_int8.to(torch.int32)


# After transposing our weights to multiply the weight and input matrix, we set that as our "accumulator", which we will add to in later steps.
accumulator = X_int32 @ w1_int32


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

print(m)

hidden_int8 = (
    ((accumulator * m + (1 << (shift - 1))) >> shift).clamp(0, 127).to(torch.int8)
)

scores = (hidden_int8.to(torch.int32) @ w2_int8.to(torch.int32)).argmax(1)
print(scores)
