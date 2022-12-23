import numpy as np
import matplotlib.pyplot as plt
import torch

from dataclasses import dataclass


import IPython


class FeedforwardMultiLayerOneDim(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_filter = 'none', 
        activation_type = "sigmoid", device = torch.device("cpu")):
        super(FeedforwardMultiLayerOneDim, self).__init__()
       
        if output_filter not in ['none', 'logistic']:
            raise ValueError("The output filter option is not recognized")


        self.output_filter = output_filter
        if output_filter == "none":
            self.output_filter = torch.nn.Identity()
        else:
            self.output_filter = torch.nn.Sigmoid()            


        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if activation_type == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        else:
            raise ValueError("Unrecognized activation type.")

        ### Create the layers
        self.layers = torch.nn.ModuleList()
        self.layers = self.layers.append(torch.nn.Linear(self.input_size, self.hidden_sizes[0]))
        for i in range(len(self.hidden_sizes)-1):
            self.layers.append(torch.nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))


        self.layers.append(torch.nn.Linear(self.hidden_sizes[-1], 1)) ### last output

        self.layers.to(device)

        #IPython.embed()


    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0 )


    # def restart_optimizer(self):
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0 )


    def forward(self, x):
        representation = x
        for i in range(len(self.hidden_sizes)):
            representation = self.layers[i](representation)
            representation = self.activation(representation)




        output = self.layers[-1](representation)
        output = torch.squeeze(output)

        output = self.output_filter(output)


        return output, representation



class FeedforwardMultiLayerOneDimMahalanobis(FeedforwardMultiLayerOneDim):
    def __init__(self, input_size, hidden_sizes, output_filter = 'none', 
        activation_type = "sigmoid", 
        device = torch.device("cpu")):
        super(FeedforwardMultiLayerOneDimMahalanobis, self).__init__(
            input_size, hidden_sizes, output_filter, 
            activation_type , device )


    def forward(self, x, inverse_data_covariance=[], alpha=0):
        representation = x
        for i in range(len(self.hidden_sizes)):
            representation = self.layers[i](representation)
            representation = self.activation(representation)




        output = self.layers[-1](representation)
        opt_output = torch.squeeze(output)
        pess_output = torch.squeeze(output)

        ### This uncertainty estimator is before the final activation function
        ### is evaluated.
        uncertainty = 0

        if len(inverse_data_covariance) != 0:
            # if torch.cuda.is_available():
            #     inverse_data_covariance = inverse_data_covariance.float().to('cuda')


            uncertainty = alpha * self.hidden_sizes[-1] * torch.sqrt(
                torch.matmul(
                    representation,
                    torch.matmul(inverse_data_covariance.float(), representation.t()),
                ).diag()
            )

            opt_output = torch.squeeze(output) + uncertainty
            pess_output = torch.squeeze(output) - uncertainty


        optimistic_predictions = self.output_filter(opt_output)
        pessimistic_predictions = self.output_filter(pess_output)

        return optimistic_predictions, representation, pessimistic_predictions

















########### Model Implementations ###############








class TorchMultilayerRegression:
    def __init__(
        self,
        dim=None,
        representation_layer_sizes=[20,10],
        activation_type = 'sigmoid',
        output_filter = 'none',
        device = torch.device("cpu")):

        if output_filter not in ['none', 'logistic']:
            raise ValueError("The output filter option is not recognized")

        self.output_filter = output_filter

        self.representation_layer_sizes = representation_layer_sizes
        self.activation_type = activation_type

        self.criterion_l2 = torch.nn.MSELoss()
        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_logistic = torch.nn.BCELoss()

        self.device = device


        if dim == None:
            raise ValueError("Dataset dimension is none")

        self.dim = dim

        self.network = FeedforwardMultiLayerOneDim(self.dim, self.representation_layer_sizes, 
            output_filter = output_filter,
            activation_type = activation_type, 
            device = device)

        self.network.to(self.device)
       

        self.restart_optimizer()

    def restart_optimizer(self):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01, weight_decay=0.0 )

    # def restart_optimizer(self):
    #      self.network.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0 )


    def reset_weights(self):
        for c in self.network.modules():
            if hasattr(c, 'reset_parameters'):
                c.reset_parameters()
                print("Reset parameters")

    def get_representation(self, batch_X):
        pred_tuple = self.network(batch_X.float())
        representations = pred_tuple[1]

        return representations

    def get_loss(self, batch_X, batch_y):
        #a = "asdflkm"
        #IPython.embed()


        # if torch.cuda.is_available():
        #     self.network.to('cuda')
        pred_tuple = self.network(batch_X.float())  # .squeeze()
        prob_predictions = pred_tuple[0]

        if self.output_filter == "logistic":
            return self.criterion_logistic(
                torch.squeeze(prob_predictions), torch.squeeze(batch_y.float())
            )
        else:
            

            #IPython.embed()
            return self.criterion_l2(
                torch.squeeze(prob_predictions), torch.squeeze(batch_y.float())
            )





    def predict(self, batch_X):
        pred_tuple = self.network(
            batch_X.float()
            )  

        prob_predictions = pred_tuple[0]
        return torch.squeeze(prob_predictions)


    def get_thresholded_predictions(self, batch_X, threshold):
        

        ### REVIVE THIS ####
        # if self.output_filter != "logistic":
        #     raise ValueError("Output filter not set to logistic")

        
        prob_predictions = self.predict(batch_X)
        thresholded_predictions = prob_predictions > threshold
        return thresholded_predictions

    def get_accuracy(self, batch_X, batch_y, threshold):
        # if self.output_filter != "logistic":
        #     raise ValueError("Output filter not set to logistic")


        thresholded_predictions = self.get_thresholded_predictions(
            batch_X, threshold)

        boolean_predictions = thresholded_predictions == batch_y.squeeze()

        #IPython.embed()

        return (boolean_predictions * 1.0).mean()


class TorchMultilayerRegressionMahalanobis(TorchMultilayerRegression):
    def __init__(
        self,
        dim=None,
        alpha=1,
        representation_layer_sizes=[20,10],
        activation_type = 'sigmoid',
        output_filter = 'none',
        device = torch.device("cpu")):


        super(TorchMultilayerRegressionMahalanobis, self).__init__(
            dim = dim, 
            representation_layer_sizes = representation_layer_sizes,
            activation_type = activation_type,
            output_filter = output_filter,
            device = device)


        if output_filter not in ['none', 'logistic']:
            raise ValueError("The output filter option is not recognized")

        self.output_filter = output_filter

        self.alpha = alpha

        self.network = FeedforwardMultiLayerOneDimMahalanobis(self.dim, self.representation_layer_sizes, 
            output_filter = output_filter, activation_type = activation_type, device = device)

        self.network.to(self.device)


        #self.optimizer= torch.optim.Adam(self.network.parameters(), lr=0.01, weight_decay=0.0 )

    def __inverse_covariance_norm(self, batch_X, inverse_covariance):
        square_norm = np.dot(np.dot(batch_X, inverse_covariance), np.transpose(batch_X))
        return np.diag(np.sqrt(square_norm))



    def predict_prob(self, batch_X, inverse_data_covariance=[]):
        prob_predictions, _, _ = self.network(
            batch_X.float(),
            inverse_data_covariance=inverse_data_covariance,
            alpha=self.alpha,
        )  
        return torch.squeeze(prob_predictions)
  
    def predict_prob_with_uncertainties(self, batch_X, inverse_data_covariance=[]):
        prob_predictions, _, pess_predictions = self.network(
            batch_X.float(),
            inverse_data_covariance=inverse_data_covariance,
            alpha=self.alpha,
        )  
        return torch.squeeze(prob_predictions), torch.squeeze(pess_predictions)



    def get_thresholded_predictions(self, batch_X, threshold, inverse_data_covariance=[]):
        prob_predictions = self.predict_prob(batch_X, inverse_data_covariance)
        thresholded_predictions = prob_predictions > threshold
        return thresholded_predictions


    def get_all_predictions_info(self, batch_X, threshold, inverse_data_covariance=[]):
        opt_prob_predictions, pess_prob_predictions = self.predict_prob_with_uncertainties(batch_X, inverse_data_covariance)
        thresholded_predictions = opt_prob_predictions > threshold
        return thresholded_predictions, opt_prob_predictions, pess_prob_predictions


    def get_accuracy(self, batch_X, batch_y, threshold, inverse_data_covariance=[]):
        thresholded_predictions = self.get_thresholded_predictions(
            batch_X, threshold, inverse_data_covariance
        )
        boolean_predictions = thresholded_predictions == batch_y
        return (boolean_predictions * 1.0).mean()















