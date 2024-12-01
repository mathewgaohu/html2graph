import torch
from torch_geometric.nn import GAE, VGAE, GCNConv, SAGEConv, GATConv, SGConv, APPNP

cached = False  # cached only for transductive learning


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=cached)  # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=cached)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# define the feature decoder
class GCNDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNDecoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv3 = GCNConv(in_channels, 2 * in_channels, cached=cached)  # cached only for transductive learning
        self.conv4 = GCNConv(2 * in_channels, out_channels, cached=cached)

    def forward(self, x, edge_index):
        x = self.conv3(x, edge_index).relu()
        return self.conv4(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VGAEDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAEDecoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        # self.conv_logstd2 = GCNConv(in_channels, 2*in_channels)
        # self.conv_mu2 = GCNConv(in_channels, 2*in_channels)
        # change from conv to dense layer
        self.l1 = torch.nn.Linear(in_channels, 2 * in_channels)
        self.l2 = torch.nn.Linear(2 * in_channels, out_channels)
        # self.conv2 = GCNConv(2* in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.l1(x).relu()
        return self.l2(x)


class GATEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATEncoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv1 = GATConv(in_channels, 2 * out_channels, cached=cached)  # cached only for transductive learning
        self.conv2 = GATConv(2 * out_channels, out_channels, cached=cached)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# define the feature decoder
class GATDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATDecoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv3 = GATConv(in_channels, 2 * in_channels, cached=cached)  # cached only for transductive learning
        self.conv4 = GATConv(2 * in_channels, out_channels, cached=cached)

    def forward(self, x, edge_index):
        x = self.conv3(x, edge_index).relu()
        return self.conv4(x, edge_index)


class SAGEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEEncoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv1 = SAGEConv(in_channels, 2 * out_channels, cached=cached)  # cached only for transductive learning
        self.conv2 = SAGEConv(2 * out_channels, out_channels, cached=cached)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# define the feature decoder
class SAGEDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGEDecoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv3 = SAGEConv(in_channels, 2 * in_channels, cached=cached)  # cached only for transductive learning
        self.conv4 = SAGEConv(2 * in_channels, out_channels, cached=cached)

    def forward(self, x, edge_index):
        x = self.conv3(x, edge_index).relu()
        return self.conv4(x, edge_index)


class SGCEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGCEncoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv1 = SGConv(in_channels, 2 * out_channels, cached=cached)  # cached only for transductive learning
        self.conv2 = SGConv(2 * out_channels, out_channels, cached=cached)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# define the feature decoder
class SGCDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGCDecoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv3 = SGConv(in_channels, 2 * in_channels, cached=cached)  # cached only for transductive learning
        self.conv4 = SGConv(2 * in_channels, out_channels, cached=cached)

    def forward(self, x, edge_index):
        x = self.conv3(x, edge_index).relu()
        return self.conv4(x, edge_index)


class APPNPEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(APPNPEncoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv1 = APPNP(in_channels, 2 * out_channels, cached=cached)  # cached only for transductive learning
        self.conv2 = APPNP(2 * out_channels, out_channels, cached=cached)  # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


# define the feature decoder
class APPNPDecoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(APPNPDecoder, self).__init__()  # super to inheret all functions of torch.nn.Module
        self.conv3 = APPNP(in_channels, 2 * in_channels, cached=cached)  # cached only for transductive learning
        self.conv4 = APPNP(2 * in_channels, out_channels, cached=cached)

    def forward(self, x, edge_index):
        x = self.conv3(x, edge_index).relu()
        return self.conv4(x, edge_index)

# decoder defaults to torch_geometric.nn.models.InnerProductDecoder
