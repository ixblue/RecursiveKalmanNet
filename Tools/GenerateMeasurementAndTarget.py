import torch
import os

def get_matrix(matrix, t, *args, **kwargs):
    """Retrieve the matrix for the current time step t, with optional extra arguments, i (batch index) for example"""
    return matrix(t, *args, **kwargs) if callable(matrix) else matrix

def load_dynamical_system(dynamical_system):
    """
    Return the list [F, Q, H, R, G] from dynamoical_system, G being None if not provided.
    Allow easy code compatibility
    """

    if len(dynamical_system) == 5:
        F, Q, H, R, G = dynamical_system
        print("Trajectory generation with control input.")
    elif len(dynamical_system) == 4:
        F, Q, H, R = dynamical_system
        G = None
    else:
        raise ValueError("Dynamical system must contain 2 or 3 elements (F, H[, G])")
    return [F, Q, H, R, G]


def generate_sequence_batched(dynamical_system, x0, P0, process_noises, measurement_noises, control):
    """
    Generate batched time series of measurements and target states.
    Args:
        dynamical_system: The matrices defining the dynamical system (F, Q, H, R, [G ]).
        x0: Initial state (state_dim, 1).
        P0: Initial covariance matrix (state_dim, state_dim).
        process_noises: Pre-generated process noises (batch_size, len_seq, state_dim, 1).
        measurement_noises: Pre-generated measurement noises (batch_size, len_seq, obs_dim, 1).
        control: Control input (batch_size, control_dim, len_seq) or None if no control input.
    Returns:
        measurements
        target states
        control_noisy: Control with noise, if G is None, None is returned.
    """
    F, Q, H, R, G = dynamical_system
    batch_size, _, len_seq = process_noises.size()
    state_dim = x0.size()[0]
    obs_dim = measurement_noises.size(1)
    if G is not None:
        control_dim = G(0).size()[1]

    # Initialize tensors for input and target states
    measure = torch.zeros(batch_size, obs_dim, len_seq)
    target = torch.zeros(batch_size, state_dim, len_seq)
    if G is not None:
        control_noisy = torch.zeros(batch_size, control_dim, len_seq)

    # Initialize the current state for all batches
    current_state = x0.unsqueeze(0).repeat(batch_size, 1, 1) + torch.distributions.MultivariateNormal(torch.zeros(state_dim), P0).sample((batch_size,)).unsqueeze(2)

    for t in range(len_seq):
        F_t, H_t = map(lambda m: get_matrix(m, t), [F, H])
        process_noise = process_noises[:, :, t].unsqueeze(-1)                           # (batch, state_dim, 1)
        measurement_noise = measurement_noises[:, :, t].unsqueeze(-1)                   # (batch, obs_dim, 1)
        control_t = control[:, :, t].unsqueeze(-1) if control is not None else None     # (batch, control_dim, 1)

        # Perform Kalman step in a batched manner, depending on whether G is provided
        if G is None:
            current_state = F_t @ current_state + process_noise
        else:
            G_t = get_matrix(G, t)
            control_noisy[:, :, t] = (control_t + process_noise).squeeze(-1)
            current_state = F_t @ current_state + G_t @ (control_t + process_noise)
            
        z = H_t @ current_state + measurement_noise

        # Save results
        target[:, :, t] = current_state.squeeze(2)
        measure[:, :, t] = z.squeeze(2)
        if G is None:
            control_noisy = None

    return measure, target, control_noisy
    


def _sample_noises_from_R_and_Q_matrices(size, len_seq, dynamical_system, noise_control_std=None):
    """
    Function to sample the process and measurement noises from the Q and R matrices of the dynamical system.
    Returns:
        process_noises : sample of the process noise (size, state_dim, len_seq)
        measurement_noises : sample of the measurement noise (size, obs_dim, len_seq)
        process_noise_covs : covariance matrix used for process noise sampling (size, state_dim, state_dim, len_seq)
        measurement_noise_covs : covariance matrix used for measurement noise sampling (size, obs_dim, obs_dim, len_seq)
    """
    F, Q, H, R, G = dynamical_system

    Q_t = get_matrix(Q, 0, 0)
    R_t = get_matrix(R, 0, 0)
    state_dim = Q_t.size()[0]
    obs_dim = R_t.size()[0]

    process_noise_means = torch.zeros(size, len_seq, state_dim)
    measurement_noise_means = torch.zeros(size, len_seq, obs_dim)
    process_noise_covs = torch.zeros(size, len_seq, state_dim, state_dim)
    measurement_noise_covs = torch.zeros(size, len_seq, obs_dim, obs_dim)

    for i in range(size):
        for t in range(len_seq):
            process_noise_means[i, t] = torch.zeros(state_dim)
            measurement_noise_means[i, t] = torch.zeros(obs_dim)
            process_noise_covs[i, t] = get_matrix(Q, i, t)
            measurement_noise_covs[i, t] = get_matrix(R, i, t)

    if G is not None:
        control_dim = get_matrix(G, 0).size()[1]
        process_noises = torch.distributions.MultivariateNormal(
            torch.zeros(size, len_seq, control_dim).view(-1, control_dim),
            noise_control_std * torch.ones(size, len_seq, control_dim).view(-1, control_dim, control_dim)
        ).sample().view(size, len_seq, control_dim) 
    else:
        process_noises = torch.distributions.MultivariateNormal(
            process_noise_means.view(-1, state_dim),
            process_noise_covs.view(-1, state_dim, state_dim)
        ).sample().view(size, len_seq, state_dim)

    measurement_noises = torch.distributions.MultivariateNormal(
        measurement_noise_means.view(-1, obs_dim),
        measurement_noise_covs.view(-1, obs_dim, obs_dim)
    ).sample().view(size, len_seq, obs_dim)

    process_noises = process_noises.permute(0, 2, 1)
    measurement_noises = measurement_noises.permute(0, 2, 1)

    return process_noises, measurement_noises, process_noise_covs.permute(0, 2, 3, 1), measurement_noise_covs.permute(0, 2, 3, 1)

def generate_measurements_and_target(batch_size, len_sequence, dynamical_system, x0, P0, data_path, control=None, noise_control_std=None):
    """
    Generate measurements and target states using the provided model and R_matrix (variances).
    Args:
        batch_size: Number of sequences to generate.
        len_sequence: Length of each sequence.
        dynamical_system: A list of the matrices defining the dynamical system (F, Q, H, R, [G ]). 
        x0: Initial state (state_dim, 1).
        P0: Initial covariance matrix (state_dim, state_dim).
        data_path: Path to save the generated dataset.
        control: Control input (batch_size, control_dim, len_sequence) or None if no control input.
        noise_control_std: Standard deviation of the control noise, if control is provided.
    Returns: data_path
        Saves a list containing of 4 or 5 elements (depending on whether G is provided):
        measurements, target states, control_noisy (only if G is provided), process_noise_covs, measurement_noise_covs
    """
    if os.path.exists(data_path):
        raise FileExistsError(f"A dataset with the name '{data_path}' already exists.")
    dirname = os.path.dirname(data_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)

    dynamical_system = load_dynamical_system(dynamical_system)
    G = dynamical_system[4]

    # Check coherence between dynamical_system and control variable value
    if G is not None and (control is None or noise_control_std is None):
        raise ValueError("Control input and control noise std is required when the dynamical system has a control matrix G.")

    process_noises, measurement_noises, process_noise_covs, measurement_noise_covs = _sample_noises_from_R_and_Q_matrices(batch_size, len_sequence, dynamical_system, noise_control_std)
    measure, target, control_noisy = generate_sequence_batched(dynamical_system, x0, P0, process_noises, measurement_noises, control)

    if G is not None:
        torch.save([measure, target, control_noisy, process_noise_covs, measurement_noise_covs], data_path)
    else:
        torch.save([measure, target, process_noise_covs, measurement_noise_covs], data_path)

    return data_path