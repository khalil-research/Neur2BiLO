import torch.nn as nn
import collections

import torch
import torch.nn as nn


class PrintLayer(nn.Module):
    """ PrintLayer class for debugging. """
    def __init__(self, printx=False):
        super(PrintLayer, self).__init__()
        self.printx = printx

    def forward(self, x):
        # Do your print / debug stuff here
        if self.printx:
            print(x)
        print(x.size(), torch.mean(x).item(), torch.count_nonzero(x, dim=1))
        return x


class FeedForwardBase(nn.Module):
    """ Standard feed forward network, with some slight Gurobi specific functions. """
    def __init__(self, input_dim, hidden_dims, output_dim, output_relu=False, dropout=0.0, bias=True, name="net"):
        """ Constructor for feed-forward net. """
        super(FeedForwardBase, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.output_relu = output_relu
        self.dropout = dropout
        self.bias = bias
        self.name = name

        layers = collections.OrderedDict()
        layers[f"{self.name}_in"] = nn.Linear(input_dim, hidden_dims[0], bias=self.bias)
        layers[f"{self.name}_act_in"] = nn.ReLU()
        
        if len(hidden_dims) == 1:
            if self.dropout:
               layers[f"{self.name}_drop_in"] = nn.Dropout(self.dropout)
            
        else:
            for i in range(len(hidden_dims) - 1):
                layers[f"{self.name}_{i}"] = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                layers[f"{self.name}_act_{i}"] = nn.ReLU()
                if self.dropout:
                    layers[f"relu_drop_in"] = nn.Dropout(self.dropout)

        if output_relu:
            layers[f"{self.name}_out"] = nn.Linear(hidden_dims[-1], output_dim, bias=self.bias)
            layers[f"{self.name}_out_relu"] = nn.ReLU()
        else:
            layers[f"{self.name}_out"] = nn.Linear(hidden_dims[-1], output_dim, bias=self.bias)


        self.layers = layers
        self.net = torch.nn.Sequential(self.layers)


    def forward(self, x):
        """ Forward for feed-forward net. """
        out = self.net(x)
        return out


    def get_grb_net(self):
        """ Gets gurobi compatible nn.sequential.  Specifically,
                - Remove dropout
                - Adds bias of 0's if no bias in layer
        """
        grb_layers = collections.OrderedDict()
        
        for name, layer in self.layers.items():

            # add ReLU layers to list
            if type(layer) == torch.nn.modules.activation.ReLU:
                grb_layers[name] = layer

            # remove dropout layer by skipping
            elif type(layer) == torch.nn.modules.dropout.Dropout:
                continue

            # linear layer
            if type(layer) == torch.nn.modules.linear.Linear:

                # if layer does not have bias, then copy weights and set bias to zero
                if layer.bias is None:
                    layer_cp = nn.Linear(layer.in_features, layer.out_features)
                    layer_cp.weight = layer.weight
                    layer_cp.bias = nn.Parameter(torch.zeros(layer.out_features))

                    grb_layers[name] = layer_cp
                    
                 # if layer has bias, then do nothing
                else:
                    grb_layers[name] = layer

        net = torch.nn.Sequential(grb_layers)
        net = net.cpu()
        net = net.eval()
            
        return net




class FeedForwardNetwork(nn.Module):
    """ Autoencoder...  """
    def __init__(self, feedforward_net, use_coef):
        """ Constructor for standard FeedForward net. """
        super(FeedForwardNetwork, self).__init__()
        self.feedforward_net = feedforward_net
        # self.sigmoid = nn.Sigmoid()
        self.use_coef = use_coef


    def forward(self, x, p):
        """ Forward for auto-encoder net. """
        # pass output through network
        out = self.feedforward_net(x)

        # dot product of ouput with coefficients
        if self.use_coef:
            out = torch.sum(torch.mul(out, p), dim=1)

        return out




class SetBasedNetwork(nn.Module):
    """ SetBasedNetwork network for variable decision size.  """
    def __init__(self, decision_embedder, value_predictor, agg_type, use_coef):
        """ Constructor for SetBasedNetwork net. """
        super(SetBasedNetwork, self).__init__()
        self.decision_embedder = decision_embedder
        self.value_predictor = value_predictor
        self.agg_type = agg_type
        self.use_coef = use_coef


    def forward(self, x, p, fs_size = None):
        """ Forward for auto-encoder net. """
        # embed upper-level decisions/information
        out = self.decision_embedder(x)
        out = self.aggregate(out, self.agg_type, fs_size)
    
        # value function prediction
        out = self.value_predictor(out)

        # if not multiplying by costs, then return
        if not self.use_coef:
            return out

        # otherwise, multiply element-wise by p
        out = torch.sum(torch.mul(out, p), dim=1)

        return out

    def aggregate(self, x, agg_type, fs_size = None):
        """ Aggregates tensors output from network. """
        if agg_type == "mean":
            if fs_size is None: # set default to being size of tensor
                fs_size = x.shape[1]

            # take true mean by summing, then dividing by first-stage dimensions
            x = torch.sum(x, axis=1)
            x = torch.div(x, fs_size.unsqueeze(1))

        elif agg_type == "sum":
            x = torch.sum(x, 1)

        return x



class SetInstanceEncodingNetwork(nn.Module):
    """   Set based instance endoging network.  """
    def __init__(self, instance_decision_embedder, final_instance_embedder, value_predictor, agg_type, use_coef, problem, approx_type):
        """ Constructor for SetBasedNetwork net. """
        super(SetInstanceEncodingNetwork, self).__init__()
        self.instance_decision_embedder = instance_decision_embedder
        self.final_instance_embedder = final_instance_embedder

        self.value_predictor = value_predictor # per decision value predictor

        self.agg_type = agg_type
        self.use_coef = use_coef

        self.problem = problem
        self.approx_type = approx_type


    def forward(self, x_inst_features, x_decisions_features, x_decision, p, fs_size = None, print_embedding = False):
        """   """
        # embed instance information
        x_inst_embedding = self.instance_decision_embedder(x_inst_features)
        x_inst_embedding = self.aggregate(x_inst_embedding, self.agg_type, fs_size)
        x_inst_embedding = self.final_instance_embedder(x_inst_embedding)
        # x_inst_embedding = 0 * x_inst_embedding # do not do this!
        if print_embedding:
            print(x_inst_embedding)

        # remove final singleton dimenaions
        x_inst_embedding = x_inst_embedding.reshape(x_inst_embedding.shape[0], x_inst_embedding.shape[1])

        # broadcast instance to number of decisions
        x_inst_embedding = x_inst_embedding[:, None, :].repeat(1, x_inst_features.shape[1], 1)

        # concatenate decision features and instance embedding
        x_decisions_features = torch.cat([x_decisions_features, x_inst_embedding], axis=2)

        # compute (1-x) * features
        # only used for kp or general binary interdiction problems
        if "kp" in self.problem:
            # broadcast decision to feature space
            x_decision = x_decision[:, :, None].repeat(1, 1, x_decisions_features.shape[-1])
            x_decisions_features = torch.mul(1 - x_decision, x_decisions_features)

        # value function prediction
        pred_per_decision = self.value_predictor(x_decisions_features)

        # reshape to exclude singleton dimension
        pred_per_decision = pred_per_decision.reshape(pred_per_decision.shape[0], pred_per_decision.shape[1])

        ## Problem-specific final computation

        # for knapsack or general linear coefficients
        if "kp" in self.problem:
            out = torch.sum(torch.mul(pred_per_decision, p), dim=1)

        # for dr, todo: potentially think about adding a bias term
        if "dr" in self.problem:
            # out = torch.sum(torch.mul(pred_per_decision, p), dim=1)
            if self.approx_type == "upper":
                out = torch.sum(torch.mul(pred_per_decision, p[:,:,0]), dim=1)
            elif self.approx_type == "lower":
                # compute v0 * (Br - pred @ c)
                pred_1 = torch.sum(torch.mul(pred_per_decision, p[:,:,0]))
                pred_2 = p[:,0,1] * (p[:,0,3] - torch.sum(torch.mul(pred_per_decision, p[:,:,2])))
                out = pred_1 + pred_2

        # for critical node game or other bilinear coefficeints
        # note that other bilinear problems will likely need to be modified depending on follower objective
        # todo: predicting output dim of (n,2) and multiplying might actually make more sense here.
        elif "cng" in self.problem:
            if self.approx_type == "lower":
                pred_1 = torch.sum(torch.mul(1 - pred_per_decision, p[:,0,:]), dim=1)
                pred_2 = torch.sum(torch.mul(pred_per_decision, p[:,1,:]), dim=1)
                pred_3 = torch.sum(torch.mul(pred_per_decision, p[:,2,:]), dim=1)
                out = pred_1 + pred_2 + pred_3
            else:
                pred_1 = torch.sum(torch.mul(1 - pred_per_decision, p[:,0,:]), dim=1)
                pred_2 = torch.sum(torch.mul(pred_per_decision, p[:,1,:]), dim=1)
                pred_3 = torch.sum(torch.mul(1 - pred_per_decision, p[:,2,:]), dim=1)
                pred_4 = torch.sum(torch.mul(pred_per_decision, p[:,3,:]), dim=1)
                out = pred_1 + pred_2 + pred_3 + pred_4

        return out


    def aggregate(self, x, agg_type, fs_size = None):
        """ Aggregates tensors output from network. """
        if agg_type == "mean":
            if fs_size is None: # set default to being size of tensor
                fs_size = x.shape[1]

            # take true mean by summing, then dividing by first-stage dimensions
            x = torch.sum(x, axis=1)
            x = torch.div(x, fs_size.unsqueeze(1))

        elif agg_type == "sum":
            x = torch.sum(x, 1)

        return x




