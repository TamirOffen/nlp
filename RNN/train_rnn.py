import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example

from rnn import RNN

category_lines, all_categories = load_data()
n_categories = len(all_categories)

def category_from_output(output):
    cat_idx = torch.argmax(output).item()
    return all_categories[cat_idx]


def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return output, loss.item()


def predict(rnn, input_line):
    print(f'\n> {input_line}')
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        guess = category_from_output(output)
        print(guess)

if __name__ == "__main__":
    n_hidden = 128
    rnn = RNN(N_LETTERS, n_hidden, n_categories)

    criterion = nn.NLLLoss()
    lr = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)

    current_loss = 0
    all_losses = []
    plot_steps, print_steps = 5000, 25_000

    n_iters = 500_000
    print("started training")
    for i in range(n_iters):
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
        output, loss = train(line_tensor, category_tensor)
        current_loss += loss
        
        if (i+1) % plot_steps == 0:
            all_losses.append(current_loss / plot_steps)
            current_loss = 0
        if (i+1) % print_steps == 0:
            guess = category_from_output(output)
            correct = "CORRECT" if guess == category else f"WRONG ({category})"
            print(f'{i+1} {int((i+1)/n_iters * 100)} {loss:.3f} {line} | {guess} {correct}')

    print('Finished Training')
    model_name = 'rnn_relu.pth'
    PATH_name = f'saved_models/{model_name}'
    torch.save(rnn.state_dict(), PATH_name)
    print(f'Saved model to {PATH_name}')

    plt.figure()
    plt.plot(all_losses)
    plt.show()
