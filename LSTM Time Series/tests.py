import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from main import LSTMPredictor


if __name__ == "__main__":
    # Sine wave creation
    N = 5 # num of samples (sine waves)
    L = 10 # num of values per sample
    T = 1 # width of wave

    x = np.array(range(L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1) 
    y = np.sin(x/1.0/T).astype(np.float32)
    print(y)
    train_input = torch.from_numpy(y)
    # inputs = train_input.split(1, dim=1)
    # x = inputs[0]
    x = train_input[0:1, :-1]
    print("input:")
    print(x.shape)

    print("output:")
    model = LSTMPredictor()
    output = model(x)
    print(output)



