import torch
from torch import nn


class ParameterMatrix(nn.Module):
    def __init__(self, input_size, output_size):
        super(ParameterMatrix, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.randn(self.input_size, self.output_size))

        self.softmax = nn.Softmax(-1)

    def forward(self):
        y = self.softmax(self.weight)
        return y


# device = torch.device("cuda:0")
# A = torch.tensor([[1, 2, 3, 4],
#                   [2, 4, 6, 8],
#                   [6, 4, 2, 1],
#                   [4, 8, 4, 1]]).to(device)
# B = ParameterMatrix(3, 4).to(device)
# # C = ParameterMatrix(4, 3).to(device)
# optim_b = torch.optim.SGD(B.parameters(), lr=0.01)
# # optim_c = torch.optim.SGD(C.parameters(), lr=0.01)
#
# for i in range(2000):
#
#     loss = torch.mean((B.weight.T @ B.weight - A)**2)
#     optim_b.zero_grad()
#     # optim_c.zero_grad()
#     loss.backward()
#     optim_b.step()
#     # optim_c.step()
#     if i % 50 == 0:
#         print(i, loss.item())
#
# tes2 = (B.weight.T @ B.weight - A).cpu().detach().numpy()
# test1 = 1
