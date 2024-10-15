import torch as pt


class MyModel(pt.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyModel, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim)

    def forward(self, ais):
        emb1,emb2 = self.mlp(ais.view(-1, output_dim)).view(ais.size(0), ais.size(1), output_dim).chunk(2, dim=1)
        d = torch.matmul(emb1.squeeze(1), emb2.squeeze(1).t())
        return d

    def inference(self, points1,points2):
        ais = 
        d = self.forward(ais)
        idx1, idx2 = hungarian(d)
        return 

        # d = torch.matmul(emb1, emb2.t())
class MLP(pt.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = pt.nn.Linear(input_dim, hidden_dim)
        self.fc2 = pt.nn.Linear(hidden_dim, output_dim)
        self.relu = pt.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x