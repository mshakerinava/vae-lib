import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    @staticmethod
    def add_arguments(arg_parser, prefix=''):
        arg_parser.add_argument('--%sact-fn' % prefix, type=str, default='tanh', help='Network activation function')
        arg_parser.add_argument('--%swidths' % prefix, type=str, nargs='*', default=[], help='Width of hidden layers in MLP')

    def __init__(self, data_shape, code_size, act_fn, widths):
        super().__init__()
        self.data_shape = data_shape
        self.data_size = 1
        for d in data_shape:
            self.data_size *= d
        self.code_size = code_size
        self.act_fn = eval('F.%s' % act_fn)
        self.widths = [int(x) for x in widths] + [self.data_size]

        self.fc = nn.ModuleList([nn.Linear(in_features=self.code_size, out_features=self.widths[0])])
        for i in range(len(self.widths) - 1):
            self.fc.append(nn.Linear(in_features=self.widths[i], out_features=self.widths[i + 1]))

    def forward(self, x):
        B = x.shape[0]
        assert x.shape == (B, self.code_size)

        for i in range(len(self.fc) - 1):
            x = self.act_fn(self.fc[i](x))
        x = self.fc[-1](x)
        assert x.shape == (B, self.widths[-1])

        x = x.view(-1, *self.data_shape)
        return x
