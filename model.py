import torch
import torch.nn as nn
import torch.nn.functional as F

class Gan3DposeNet(nn.Module):

	def __init__(self, n_inputs=34, n_unit=1024, mode='generator'):
		super(Gan3DposeNet, self).__init__()

		n_outputs = n_inputs // 2 if mode == 'generator' else 1

		self.fc1 = nn.Linear(n_inputs, n_unit)
		self.fc2 = nn.Linear(n_unit, n_unit)
		self.fc3 = nn.Linear(n_unit, n_unit)
		self.fc4 = nn.Linear(n_unit, n_outputs)

	def forward(self, x):
		x1 = F.LeakyReLU(self.fc1(x))
		x2 = F.LeakyReLU(self.fc2(x1))
		x3 = F.LeakyReLU(self.fc3(x2) + x1)
		x4 = self.fc4(x3)

		return x4



