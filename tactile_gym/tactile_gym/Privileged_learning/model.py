import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
class Student(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=4, action_space=2, use_gpu=False, dropout=0.8):
        super(Student, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.rnn_1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, action_space)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.h0 = None
        #self.c0 = None

    def forward(self, x):

        if self.use_gpu:
            h0 = torch.zeros(self.num_layers, len(x), self.hidden_size).to(device)
            #c0 = torch.zeros(self.num_layers, len(x), self.hidden_size).to(device)
        else:
            h0 = torch.zeros(self.num_layers, len(x), self.hidden_size)
            #c0 = torch.zeros(self.num_layers, len(x), self.hidden_size)

        #  LSTM
        x, _ = self.rnn_1(x, h0)  # out的格式 (batch_size, seq_length, hidden_size)
        out = self.dropout_1(x)
        out = self.fc(out[:, -1, :])
        return out



if __name__ == "__main__":
    network = Student()
    print(network)
