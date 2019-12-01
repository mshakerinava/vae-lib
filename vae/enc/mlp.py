import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    @staticmethod
    def add_arguments(arg_parser, prefix=''):
        arg_parser.add_argument('--%sact-fn' % prefix, type=str, default='tanh', help='Network activation function')
        arg_parser.add_argument('--%swidths' % prefix, type=str, nargs='*', default=[], help='Width of hidden layers in MLP')

    def __init__(self, data_shape, code_size, act_fn, widths):
        super().__init__()
        self.data_size = 1
        for d in data_shape:
            self.data_size *= d
        self.code_size = code_size
        self.act_fn = eval('F.%s' % act_fn)
        self.widths = [self.data_size] + [int(x) for x in widths]

        self.fc = nn.ModuleList([])
        for i in range(len(self.widths) - 1):
            self.fc.append(nn.Linear(in_features=self.widths[i], out_features=self.widths[i + 1]))
        self.fc_mean = nn.Linear(in_features=self.widths[-1], out_features=self.code_size)
        self.fc_log_var = nn.Linear(in_features=self.widths[-1], out_features=self.code_size)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(-1, self.data_size)

        for fc in self.fc:
            x = self.act_fn(fc(x))
        assert x.shape == (B, self.widths[-1])

        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        assert mean.shape == (B, self.code_size)
        assert log_var.shape == (B, self.code_size)
        return mean, log_var
