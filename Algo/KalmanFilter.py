import torch

# Define the base Kalman Filter class
class KalmanFilter:
    def __init__(self, dynamical_system, x0, P0):
        """
        Initialize the Kalman Filter.
        dynamical_system : being a list of F, H, Q, R, G (optional) defining the dynamical system
        x0: Initial state estimate
        P0: Initial error covariance of the estimate
        """
        F, Q, H, R, G = self.load_dynamical_system(dynamical_system)

        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.G = G
        self.x = x0.unsqueeze(0)
        self.P = P0.unsqueeze(0)
        self.x0 = x0.clone()
        self.P0 = P0.clone()

        self.obs_dim = H(0).size()[0]
        self.state_dim = H(0).size()[1]
        if G is not None:
            self.control_dim = self.G(0).size()[1]

    def load_dynamical_system(self, dynamical_system):
        if len(dynamical_system) == 4:
            F, Q, H, R = dynamical_system
            G = None 
        elif len(dynamical_system) == 5:
            F, Q, H, R, G = dynamical_system
        else:
            raise ValueError("dynamical_system must contain 4 or 5 elements (F, Q, H, R[, G])")
        return F, Q, H, R, G
    
    def initialise_variables(self, batch_size):
        self.init_sequence(batch_size)

    def init_sequence(self, batch_size): 
        self.x = self.x0.unsqueeze(0).repeat(batch_size, 1, 1)  # Initial state
        self.P = self.P0.unsqueeze(0).repeat(batch_size, 1, 1)  # Initial covariance

    def get_matrix(self, matrix, t):
        """Retrieve the matrix for the current time step t."""
        mat = matrix(t) if callable(matrix) else matrix
        return mat.unsqueeze(0) if mat.dim() == 2 else mat

    def predict_state(self, t, u=None):
        """Predict the next state (x)."""
        self.F_t = self.get_matrix(self.F, t)
        if self.G is None:
            self.x_prior = self.F_t @ self.x
        else:
            self.G_t = self.get_matrix(self.G, t)            
            self.x_prior = self.F_t @ self.x + self.G_t @ u

    def predict_cov(self, t):
        """Predict the next error covariance (P)."""
        batch_size = self.x.size(0)
        Q_t_list = []
        for i in range(batch_size):
            Q_i = self.Q(i, t)
            Q_t_list.append(Q_i.unsqueeze(0))
        self.Q_t = torch.cat(Q_t_list, dim=0)  # (batch_size, state_dim, state_dim)

        self.P_prior = self.F_t @ self.P @ self.F_t.transpose(1, 2) + self.Q_t

    def calc_innov(self, z, t):
        """Calculate the measurement residual (y) and store H_t as an instance variable."""
        self.H_t = self.get_matrix(self.H, t)
        self.y = z - self.H_t @ self.x_prior

    def calc_innov_cov(self, t):
        """Calculate the innovation covariance (S) and store R_t as an instance variable."""
        batch_size = self.x.size(0)
        obs_dim = self.H_t.size(1)
        # Build R_t for each batch element using self.R(i, t)
        R_t_list = []
        for i in range(batch_size):
            R_i = self.R(i, t)
            R_t_list.append(R_i.unsqueeze(0))
        self.R_t = torch.cat(R_t_list, dim=0)  # (batch_size, obs_dim, obs_dim)
        self.S = self.H_t @ self.P_prior @ self.H_t.transpose(1, 2) + self.R_t

    def calc_gain(self):
        """Calculate the Kalman Gain."""
        self.K = self.P_prior @ self.H_t.transpose(1, 2) @ torch.linalg.inv(self.S)

    def update_cov(self):
        """Update error covariance."""
        I = torch.eye(self.P.shape[1], device=self.P.device).unsqueeze(0)
        IKH = I - self.K @ self.H_t

        # Ensure batch dimensions for F_t, Q_t, and R_t
        self.F_t = self.F_t.unsqueeze(0) if self.F_t.dim() == 2 else self.F_t
        self.Q_t = self.Q_t.unsqueeze(0) if self.Q_t.dim() == 2 else self.Q_t
        self.R_t = self.R_t.unsqueeze(0) if self.R_t.dim() == 2 else self.R_t

        self.P = IKH @ self.P_prior @ IKH.transpose(1, 2) + self.K @ self.R_t @ self.K.transpose(1, 2)

    def update_state(self):
        """Update the state estimate."""
        self.x = self.x_prior + self.K @ self.y

    def step(self, z, t, u):
        """Perform a single time step of the Kalman Filter process."""
        self.predict_state(t, u)
        self.predict_cov(t)
        self.calc_innov(z, t)
        self.calc_innov_cov(t)
        self.calc_gain()
        self.update_state()
        self.update_cov()

    def process_batch(self, measure, control=None):
        """
        Process a batch of time series data using the Kalman filter.
        Args:
            measure: Measurements tensor of shape (batch_size, obs_dim, len_sequence).
            control: Control tensor of shape (batch_size, control_dim, len_sequence) or None if no control
        Returns:
            states: Tensor of estimated states (batch_size, state_dim, len_sequence).
            gains: Tensor of Kalman gains (batch_size, state_dim, obs_dim, len_sequence).
            covariances: Tensor of covariance matrices of the states (batch_size, state_dim, state_dim, len_sequence).
        """
        # Control coherence check, ensure both G and control are either provided or not
        if (self.G is not None and control is None) or (self.G is None and control is not None):
            raise ValueError("Control must be provided if and only if G is defined in the dynamical system.")
        
        # Size shortcuts
        batch_size, obs_dim, len_sequence = measure.size()
        state_dim = self.x.size(1)

        # Initialize tensors to store results
        states = torch.zeros(batch_size, state_dim, len_sequence, device=measure.device)
        gains = torch.zeros(batch_size, state_dim, obs_dim, len_sequence, device=measure.device)
        covariances = torch.zeros(batch_size, state_dim, state_dim, len_sequence, device=measure.device)

        # Initialize varaibles for the batch
        self.initialise_variables(batch_size)

        # Iterate over time steps
        for t in range(len_sequence):
            z_t = measure[:, :, t].unsqueeze(2)                                     # (batch_size, obs_dim, 1)
            u_t = control[:, :, t].unsqueeze(2) if control is not None else None    # (batch_size, control_dim, 1) 
            self.step(z_t, t, u_t)                                                  # Perform a single time step 

            # Save results
            states[:, :, t] = self.x.squeeze(2)  # (batch_size, state_dim)
            gains[:, :, :, t] = self.K  # (batch_size, state_dim, obs_dim)
            covariances[:, :, :, t] = self.P  # (batch_size, state_dim, state_dim)

        return states, gains, covariances