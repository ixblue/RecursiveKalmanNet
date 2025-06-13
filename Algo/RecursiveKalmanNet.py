import torch
import torch.nn as nn
import os
import random

from Algo.KalmanFilter import KalmanFilter
from Algo.LossFunctions import MSELoss, MAPELoss, GaussianLikelihoodLoss

class RKN(KalmanFilter, nn.Module):
    def __init__(self, dynamical_system, x0, P0, model_name, weight_factor=1, GRU_Network_configuration=None):
        """
        Initialize the RKN class, inheriting from BaseKalmanFilter and nn.Module.
        Args:
            dynamical_system: List containing the matrices or functions [F, H, [G ]] defining the dynamical system, without Q and R since the RKN works without them
            F: State transition matrix or function.
            x0: Initial state estimate.
            P0: Initial error covariance.
            model_name: Filename (str) for the GRUNetworks, checked in .models/ directory.
            weight_factor: Factor to divide the weights of fully connected layers after default initialization.
            GRU_Network_configuration: Dictionary containing the configuration for the GRUNetworks, 
        """
        F, H, G = self.load_dynamical_system_RKN(dynamical_system)

        nn.Module.__init__(self)
        KalmanFilter.__init__(self, [F, None, H, None, G], x0, P0)
        self.model_name = model_name
        self.weight_factor = weight_factor

        self.GRU_Network_configuration = GRU_Network_configuration
        self.load_or_init_rnns(model_name, GRU_Network_configuration)

    def load_dynamical_system_RKN(self, dynamical_system):
        if len(dynamical_system) == 3:
            F, H, G = dynamical_system
            print("RKN initialized with dynamical system containing G (control input matrix) Control variables must be specified when trained and then used.")
        elif len(dynamical_system) == 2:
            F, H = dynamical_system
            G = None
        else:
            raise ValueError("Dynamical system must contain 2 or 3 elements (F, H[, G])")
        return F, H, G


    def load_or_init_rnns(self, model_name, GRU_network_configuration=None):
        """
        Loads rnn_K and rnn_cov from .models/model_name if exists,
        otherwise initializes new GRUNetworks.
        """
        os.makedirs('.models', exist_ok=True)
        model_path = os.path.join('.models', model_name+'.pt')
        if os.path.isfile(model_path):
            rnns_loaded = torch.load(model_path, map_location='cpu')
            self.rnn_K = rnns_loaded['rnn_K']
            self.rnn_cov = rnns_loaded['rnn_cov']
            print(f"Loaded rnn_K and rnn_cov from {model_path}")
        else:
            obs_dim = self.H(0).size()[0]
            state_dim = self.H(0).size()[1]

            if GRU_network_configuration is None:
                GRU_network_configuration = {
                    'nb_layer_FC1': 1,
                    'FC1_mult': 10,
                    'nbr_GRU': 1,
                    'hidden_size_mult': 10,
                    'nb_layer_FC2': 1,
                    'FC2_mult': 20,
                }
                self.rnn_K = GRUNetwork(state_dim, obs_dim, output_size=state_dim*obs_dim, config=GRU_network_configuration, weight_factor=self.weight_factor)
                self.rnn_cov = GRUNetwork(state_dim, obs_dim, output_size=int((state_dim*(state_dim+1))/2), config=GRU_network_configuration, weight_factor=self.weight_factor)
                print(f"No RNN found at {model_path} : Initialized new GRUNetworks with default configuration")
            
            else:
                self.rnn_K = GRUNetwork(state_dim, obs_dim, output_size=state_dim*obs_dim, config=GRU_network_configuration, weight_factor=self.weight_factor)
                self.rnn_cov = GRUNetwork(state_dim, obs_dim, output_size=int((state_dim*(state_dim+1))/2), config=GRU_network_configuration, weight_factor=self.weight_factor)
                print(f"No RNN found at {model_path} : Initialized new GRUNetworks with specified configuration.")

    def initialise_variables(self, batch_size):
        self.init_sequence(batch_size)
        self.rnn_K.reset_hidden_state(batch_size)
        self.rnn_cov.reset_hidden_state(batch_size)


    def init_sequence(self, batch_size): 
        # Initialize the Kalman filter for the batch
        self.x = self.x0.unsqueeze(0).repeat(batch_size, 1, 1)  # Initial state
        self.P = self.P0.unsqueeze(0).repeat(batch_size, 1, 1)  # Initial covariance
        
        self.x_previous = self.x.clone()  # Store the previous state
        self.x_prior_previous = self.get_matrix(self.F, 0) @ self.x_previous
        self.z_previous = self.get_matrix(self.H, 0) @ self.x_previous
    

    def calc_gain(self):
        """Calculate the Kalman Gain using the GRU network."""
        K_vec = self.rnn_K(self.features)  # (1, batch_size, state_dim*obs_dim)

        # Reshape K_vec to (batch_size, state_dim, obs_dim)
        batch_size = K_vec.size(1)
        state_dim = self.state_dim
        obs_dim = self.obs_dim
        self.K = K_vec.view(batch_size, state_dim, obs_dim)

    def update_cov(self):
        """Update error covariance using the GRU network."""
        self.C = self.rnn_cov(self.features) # (1, batch_size, state_dim*(state_dim+1)/2)

        # Compute the first part of the Covariance matrix, refered as A in the paper
        I = torch.eye(self.P.shape[1], device=self.P.device).unsqueeze(0)
        IKH = I - self.K @ self.H_t
        self.A = IKH @ self.F_t @ self.P @ self.F_t.transpose(1,2) @ IKH.transpose(1, 2)

        # Convert the output to a symmetric positive-definite matrix, refered as B in the paper
        tril_indices = torch.tril_indices(self.x.size(1), self.x.size(1))
        L = torch.zeros(self.x.size(0), self.x.size(1), self.x.size(1), device=self.x.device)
        L[:, tril_indices[0], tril_indices[1]] = self.C
        self.B = L @ L.transpose(1, 2)  # P = L * L^T

        self.P = self.A + self.B

    def step(self, z, t, u):
        """Perform a single step of the RKN process."""
        self.predict_state(t, u)
        self.calc_innov(z, t)
        self.get_features(z)
        self.calc_gain()
        self.update_state()
        self.update_cov()
        

    def get_features(self, z):
        # Innovation (Already computed in the step() function): self.y 
        # Correction on t-1 : 
        correction = self.x - self.x_prior_previous
        # Pbservation difference : 
        z_diff = z - self.z_previous

        # Update previous values
        self.z_previous = z 
        self.x_prior_previous = self.x_prior

        # Reshape features into a 1D array
        self.features = torch.cat([
            self.y.squeeze(-1),
            z_diff.squeeze(-1),
            self.H_t.squeeze(0).repeat(z_diff.size()[0], 1),
            correction.squeeze(-1)
        ], dim=1).unsqueeze(0)  # Shape : (1, batch_size, nb_features) To fit in the GRU network

        self.features = torch.square(self.features)
    


class GRUNetwork(nn.Module):
    def __init__(self, state_dim, obs_dim, output_size, config, weight_factor=1):
        super(GRUNetwork, self).__init__()
        """
        Initialize the GRUNetwork class.
        Args:
            state_dim: Dimension of the state vector.
            obs_dim: Dimension of the observation vector.
            output_size: Size of the output layer.
            config: Dictionary containing configuration parameters:
                - nb_layer_FC1: Number of fully connected layers before GRU.
                - FC1_mult: Multiplier for the number of neurons in the fully connected layers.
                - nbr_GRU: Number of GRU layers.
                - hidden_size_mult: Multiplier for the hidden size of the GRU.
                - nb_layer_FC2: Number of fully connected layers after GRU.
                - FC2_mult: Multiplier for the number of neurons in the fully connected layers after GRU.
            weight_factor: Factor to divide the weights of fully connected layers after default initialization.
        """

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.output_size = output_size

        nb_layer_FC1 = config['nb_layer_FC1']
        FC1_mult = config['FC1_mult']
        self.nbr_GRU = config['nbr_GRU']
        self.hidden_size_mult = config['hidden_size_mult']
        nb_layer_FC2 = config['nb_layer_FC2']
        FC2_mult = config['FC2_mult']

        self.input_size = 2 * self.obs_dim + self.state_dim * self.obs_dim + self.state_dim
        self.hidden_size = self.output_size * self.hidden_size_mult

        # Fully connected layers before GRU
        fc1_layers = []
        for i in range(nb_layer_FC1):
            in_features = self.input_size if i == 0 else self.input_size * FC1_mult
            out_features = self.input_size * FC1_mult
            fc1_layers.append(nn.Linear(in_features, out_features))
            fc1_layers.append(nn.ReLU())
        self.fc1 = nn.Sequential(*fc1_layers)

        # GRU layer
        self.gru = nn.GRU(self.input_size * FC1_mult, self.hidden_size, self.nbr_GRU)

        # Fully connected layers after GRU
        fc2_layers = []
        for i in range(nb_layer_FC2):
            in_features = self.hidden_size if i == 0 else self.hidden_size * FC2_mult
            out_features = self.hidden_size * FC2_mult
            fc2_layers.append(nn.Linear(in_features, out_features))
            fc2_layers.append(nn.ReLU())
        self.fc2 = nn.Sequential(*fc2_layers)

        # Final output layer
        self.output_layer = nn.Linear(self.hidden_size * FC2_mult, self.output_size)

        # GRU hidden state initialized to None
        self.hidden = None

        # Divide weights of all fully connected layers by weight_factor
        if weight_factor != 1:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data *= weight_factor
                    if m.bias is not None:
                        m.bias.data *= weight_factor

    def reset_hidden_state(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.nbr_GRU, batch_size, self.hidden_size).zero_()
        self.hidden = hidden.data

    def forward(self, x):
        x = self.fc1(x)
        x, self.hidden = self.gru(x, self.hidden)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x

class Trainer:
    def __init__(self, model, learning_rate, use_cuda, seed=0, training_loss=None):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.best_val_loss = float('inf')
        self.best_model = None
        self.seed = seed
        self.training_loss = training_loss
        
        if training_loss is None:
            self.criterion = MSELoss()
        else:
            self.criterion = training_loss

        if use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Shortcuts
        self.n = self.model.obs_dim
        self.m = self.model.state_dim

    def get_training_batch(self, train_measure, train_target, batch_size, epoch, train_control=None):
        """
        Returns train_measure_batch, train_target_batch and train_control_batch (if control variable provided) for a random batch of the training dataset.
        If the seed is set, the random samples will be : - different for each epoch
                                                         - the same for each training run.
        """
        N_E = train_measure.size()[0]
        n = self.model.obs_dim
        m = self.model.state_dim
        T = train_measure.size()[2]

        train_measure_batch = torch.zeros([batch_size, n, T]).to(self.device)
        train_target_batch = torch.zeros([batch_size, m, T]).to(self.device)
        if train_control is not None:
            train_control_batch = torch.zeros([batch_size, self.model.control_dim, T]).to(self.device)

        if self.seed is not None:
            random.seed(epoch * self.seed)
        n_e = random.sample(range(N_E), k=batch_size)

        for i, idx in enumerate(n_e):
            train_measure_batch[i, :, :] = train_measure[idx]
            train_target_batch[i, :, :] = train_target[idx]
            if train_control is not None:
                train_control_batch[i, :, :] = train_control[idx]

        if train_control is None:
            return train_measure_batch, train_target_batch, None
        else:
            return train_measure_batch, train_target_batch, train_control_batch

    def compute_loss(self, prediction, target, cov=None):
        # Handle different loss signatures, expecting different inputs
        if isinstance(self.criterion, GaussianLikelihoodLoss):
            return self.criterion(prediction, target, cov)
        else:
            # Standard loss expects (output, target)
            return self.criterion(prediction, target)
        
    def set_decreasing_learning_rate(self, decreasing_learning_rate, initial_lr, final_lr, n_epochs, step_size=1):
        """
        Set the learning rate to decrease through epochs.
        Args:
            decreasing_learning_rate: Boolean to set the learning rate decreasing.
            initial_lr: Initial learning rate at epoch 0.
            final_lr: Final learning rate at last epoch.
            n_epochs: Number of epochs for the learning rate to decrease.
            step_size: Number of epochs between each lr update.
        """
        if decreasing_learning_rate:
            self.decreasing_learning_rate = True
            self.learning_rate = initial_lr         # Overwrite the learning rate specified in the initialization
            self.final_lr = final_lr
            self.n_epochs = n_epochs
            self.step_size = step_size
            
            gamma = (self.final_lr / self.learning_rate) ** (1 / (self.n_epochs / self.step_size))
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=gamma)
        else:
            self.decreasing_learning_rate = False

    def train(self, train_measure, train_target, cv_measure, cv_target, n_epochs, batch_size, train_control=None, cv_control=None):
        train_measure, train_target = train_measure.to(self.device), train_target.to(self.device)
        cv_measure, cv_target = cv_measure.to(self.device), cv_target.to(self.device)

        # Control variable check 
        if self.model.G is not None and (train_control is None or cv_control is None):
            raise ValueError("Control variables must be provided when the model has a control input matrix(G matrix).")

        # Shortcuts
        self.N_E = train_measure.size()[0]        
        self.N_CV = cv_measure.size()[0]     
        self.N_B = batch_size
        self.T = train_measure.size()[2]

        # Initial validation loss, allows to keep the original model in case of no improvement
        x_cv, _, cov_cv = self.model.process_batch(cv_measure, cv_control)
        self.best_val_loss = self.compute_loss(x_cv, cv_target, cov_cv)

        # Initialize losses and optimal epoch   
        self.train_losses = torch.zeros([n_epochs])
        self.cv_losses = torch.zeros([n_epochs])
        self.optimal_epoch = 0

        for epoch in range(n_epochs):
            ### TRAINING PHASE ###
            self.model.train()
            self.optimizer.zero_grad()

            # Get training batch
            train_measure_batch, train_target_batch, train_control_batch = self.get_training_batch(train_measure, train_target, self.N_B, epoch, train_control)

            # Use process_batch for batch prediction
            x_training_batch, _, cov_training_batch = self.model.process_batch(train_measure_batch, train_control_batch)

            # Compute loss
            loss = self.compute_loss(x_training_batch, train_target_batch, cov_training_batch)

            # Backward pass and optimization
            loss.backward(retain_graph = True)
            self.optimizer.step()
            self.train_losses[epoch] = loss.item()
            
            if self.decreasing_learning_rate:
                self.scheduler.step()
                print("Current learning rate : ", self.scheduler.get_last_lr()) # Print current learning rate
 

            ### VALIDATION PHASE ###
            self.model.eval()
            with torch.no_grad():
                # Use process_batch for validation prediction
                x_cv, _, cov_cv = self.model.process_batch(cv_measure, cv_control)
                cv_loss = self.compute_loss(x_cv, cv_target, cov_cv)
                self.cv_losses[epoch] = cv_loss.item()

                ## Store the best epoch and the best Loss obtained
                if (self.cv_losses[epoch] < self.best_val_loss):
                    self.best_val_loss = self.cv_losses[epoch]
                    self.optimal_epoch = epoch
                    # Save both GRUNetworks from the RKN model in one file in .models/self.model_name.pt
                    if hasattr(self.model, 'rnn_K') and hasattr(self.model, 'rnn_cov'):
                        save_path = os.path.join('.models', f"{self.model.model_name}.pt")
                        torch.save({'rnn_K': self.model.rnn_K, 'rnn_cov': self.model.rnn_cov}, save_path)

                # Save loss curves after each epoch
                loss_dir = os.path.join('.results', 'loss_curves')
                os.makedirs(loss_dir, exist_ok=True)
                loss_path = os.path.join(loss_dir, f"{self.model.model_name}.pt")
                torch.save({'train_losses': self.train_losses, 'cv_losses': self.cv_losses}, loss_path)

            ### EPOCH SUMMARY AND LOGGING ###
            print("Epoch : ", epoch, f" Training Loss : {loss.detach().item():.4f} | Validation Loss : {self.cv_losses[epoch].item():.4f}")
            print("Optimal idx :", self.optimal_epoch, f"Optimal :{self.best_val_loss.item():.4f}")  

        ### END OF TRAINING LOGGING ###
        print("Training complete. Best Validation Loss:", self.best_val_loss)
        print("Optimal model obtained at epoch:", self.optimal_epoch, " is loaded")
        self.model.load_or_init_rnns(self.model.model_name)