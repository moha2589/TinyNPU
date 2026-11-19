import numpy as np
from torchvision import datasets

PSUM_BITS = 18
ACC_BITS = 32
ARRAY_SIZE = 4

_d = np.load("golden/w_int8.npz")
W1 = _d["w1"].astype(np.int64)
W2 = _d["w2"].astype(np.int64)
M = int(_d["m"])
SHIFT = int(_d["shift"])


def wrap(value, bits):
    half = 2 ** (bits - 1)
    return ((value + half) % (2 * half)) - half


def pe_mac(psum, x, weight):
    assert -128 <= x <= 127
    assert -128 <= weight <= 127
    return wrap(psum + x * weight, PSUM_BITS)


def tile(x4, W4):
    """
    x0 -> 0 -> 0 -> 0 -> 0
          |    |    |    |
    x1 -> 0 -> 0 -> 0 -> 0
          |    |    |    |
    x2 -> 0 -> 0 -> 0 -> 0
          |    |    |    |
    x3 -> 0 -> 0 -> 0 -> 0
          v    v    v    v
         p0   p1   p2   p3

    """
    psums = [0] * ARRAY_SIZE
    for col in range(ARRAY_SIZE):
        for row in range(ARRAY_SIZE):
            psums[col] = pe_mac(psums[col], int(x4[row]), int(W4[col][row]))
    return psums


def layer(x, W):
    n = ARRAY_SIZE
    outputs = [0] * len(W)

    for a in range(len(W) // n):
        totals = [0] * n
        for b in range(len(W[0]) // n):
            W4 = W[n * a : n * a + n, n * b : n * b + n]
            x4 = x[n * b : n * b + n]
            part = tile(x4, W4)
            for k in range(n):
                totals[k] = wrap(totals[k] + part[k], ACC_BITS)
        outputs[n * a : n * a + n] = totals

    return outputs


def requantize(acc):
    y = (acc * M + (1 << (SHIFT - 1))) >> SHIFT
    return max(0, min(127, y))


def helper(W):
    extra = -len(W) % ARRAY_SIZE
    return np.vstack([W, np.zeros((extra, W.shape[1]), dtype=W.dtype)])


def network(x):
    hidden = [requantize(v) for v in layer(x, W1)]
    scores = layer(np.array(hidden), helper(W2))[: len(W2)]
    return scores, int(np.argmax(scores))


def quantize_image(pixels):
    return np.round(np.asarray(pixels, dtype=np.float64) / 255 * 127).astype(np.int64)


def test(n_images=None):
    test_data = datasets.MNIST(root="data", train=False, download=True)
    size = len(test_data) if n_images is None else n_images
    correct = 0

    for i in range(size):
        img, label = test_data[i]
        x = quantize_image(np.array(img).reshape(-1))
        _, digit = network(x)
        correct += digit == label

    print(f"Golden model ({size} images): \n Accuracy: {(100 * correct / size):>0.1f}% \n")


if __name__ == "__main__":
    test(n_images=500)
