from rnn import RNN 
import torch
from utils import N_LETTERS, load_data
from train_rnn import predict

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
category_lines, all_categories = load_data()
n_categories = len(all_categories)
n_hidden = 128
saved_rnn_model = RNN(N_LETTERS, n_hidden, n_categories)
final_model_path = '/Users/tamiroffen/Documents/nlp/RNN/saved_models/rnn_name.pth'
saved_rnn_model.load_state_dict(torch.load(final_model_path))
# # saved_rnn_model.to(device)
saved_rnn_model.eval()

while True:
    sentence = input("Input a last name:")
    if sentence == "quit":
        break
    predict(saved_rnn_model, sentence)
