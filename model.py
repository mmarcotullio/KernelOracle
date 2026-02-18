import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))
import torch.nn as nn
import utils

device = utils.DEVICE


class Sequence2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(Sequence2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTMCell(self.in_dim, self.hidden_dim)
        self.lstm2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)
        self.out_to_in = nn.Linear(self.out_dim, self.in_dim)
    def forward(self, input_tensor, future=0):
        outputs = []
        B, T, F = input_tensor.shape  # (batch, time, features)

        h_t  = torch.zeros(B, self.hidden_dim, device=input_tensor.device, dtype=input_tensor.dtype)
        c_t  = torch.zeros(B, self.hidden_dim, device=input_tensor.device, dtype=input_tensor.dtype)
        h_t2 = torch.zeros(B, self.hidden_dim, device=input_tensor.device, dtype=input_tensor.dtype)
        c_t2 = torch.zeros(B, self.hidden_dim, device=input_tensor.device, dtype=input_tensor.dtype)

        for t in range(T):
            input_t = input_tensor[:, t, :].contiguous()   # (B, F) contiguous
            h_t,  c_t  = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        for _ in range(future):
            # see note #2 below about dimension matching
            x = self.out_to_in(output)
            h_t,  c_t  = self.lstm1(x, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        return torch.stack(outputs, 1).squeeze(2)
