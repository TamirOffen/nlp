from rnn import RNN 
import torch
from utils import N_LETTERS, load_data
from train_rnn import predict


category_lines, all_categories = load_data()
n_categories = len(all_categories)
n_hidden = 128
saved_rnn_model = RNN(N_LETTERS, n_hidden, n_categories)
final_model_path = 'saved_models/rnn_relu.pth'
saved_rnn_model.load_state_dict(torch.load(final_model_path))
saved_rnn_model.eval()

while True:
    sentence = input("Input a last name:")
    if sentence == "quit":
        break
    predict(saved_rnn_model, sentence)
