import torch
import os

def obs_noise_distribution_shape_verification(
    regime_change,
    obs_noise_distribution,
    len_seq=None,
    multiple_configs=False,
    size=None,
    distribution_over_batch="constant"
):
    """
    Verify that obs_noise_distribution matches the requirements for the given regime_change value, multiple_configs, and distribution_over_batch.
    """
    def check_normal(val):
        if distribution_over_batch == "constant":
            return isinstance(val, (float, int))
        else:
            return (
                isinstance(val, (tuple, list)) and len(val) == 2 and
                all(isinstance(v, (float, int)) for v in val)
            )

    def check_bimodal(val):
        if distribution_over_batch == "constant":
            return (
                isinstance(val, (list, tuple)) and len(val) == 3 and
                all(isinstance(v, (float, int)) for v in val)
            )
        else:
            return (
                isinstance(val, (tuple, list)) and len(val) == 2 and
                all(isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (float, int)) for x in v) for v in val)
            )

    if not multiple_configs:
        if not regime_change:
            # obs_noise_distribution: [distribution, values]
            if (not isinstance(obs_noise_distribution, (list, tuple)) or len(obs_noise_distribution) != 2):
                raise ValueError("If regime_change is False, obs_noise_distribution must be [distribution, values].")
            distribution, values = obs_noise_distribution
            if distribution not in ['normal', 'bimodal']:
                raise ValueError("Distribution must be 'normal' or 'bimodal'.")
            if distribution == 'normal':
                if not check_normal(values):
                    raise ValueError(
                        "For 'normal', values must be {}.".format(
                            "a float (sigma)" if distribution_over_batch == "constant" else "a tuple/list of two floats (min, max)"
                        )
                    )
            elif distribution == 'bimodal':
                if not check_bimodal(values):
                    raise ValueError(
                        "For 'bimodal', values must be {}.".format(
                            "a list/tuple of 3 floats (sigma1, sigma2, p1)" if distribution_over_batch == "constant"
                            else "a tuple/list of two lists/tuples of 3 floats (min, max) for each parameter"
                        )
                    )
        else:
            # obs_noise_distribution: [list of distributions, list of values, list of regime change instants]
            if (not isinstance(obs_noise_distribution, (list, tuple)) or len(obs_noise_distribution) != 3):
                raise ValueError("If regime_change is True, obs_noise_distribution must be [list of distributions, list of values, list of regime change instants].")
            distributions, values_list, transitions = obs_noise_distribution
            N = len(distributions)
            if not (isinstance(distributions, (list, tuple)) and isinstance(values_list, (list, tuple)) and isinstance(transitions, (list, tuple))):
                raise ValueError("For regime_change=True, obs_noise_distribution must be [list, list, list].")
            if len(values_list) != N:
                raise ValueError("Length of values_list must match length of distributions.")
            if len(transitions) != N - 1:
                raise ValueError("Length of transitions must be N-1.")
            for i in range(N):
                # Reuse the single regime check for each regime
                obs_noise_distribution_shape_verification(
                    False, [distributions[i], values_list[i]], len_seq, False, None, distribution_over_batch
                )
            if len_seq is not None and not all(0 < t < len_seq for t in transitions):
                raise ValueError("All regime change instants must be >0 and <len_seq.")
    else:
        # multiple_configs=True: obs_noise_distribution must be [list of distributions, list of batch sizes]
        if not (isinstance(obs_noise_distribution, (list, tuple)) and len(obs_noise_distribution) == 2):
            raise ValueError("If multiple_configs=True, obs_noise_distribution must be [list of distributions, list of batch sizes].")
        dist_list, batch_sizes = obs_noise_distribution
        if not (isinstance(dist_list, (list, tuple)) and isinstance(batch_sizes, (list, tuple))):
            raise ValueError("If multiple_configs=True, obs_noise_distribution must be [list, list].")
        if size is not None and sum(batch_sizes) != size:
            raise ValueError("Sum of batch_sizes must equal size.")
        for dist in dist_list:
            obs_noise_distribution_shape_verification(regime_change, dist, len_seq, False, None, distribution_over_batch)


def generate_std_values(
    batch_size,
    len_sequence,
    regime_change=False,
    distribution_over_batch="constant",
    multiple_configs=False,
    obs_noise_distribution=None,
):
    """
    Generate the standart deviation value of R or Q for each batch and instant.
    Useful in case of data generation with different configuration over the batch.
    Args:
        batch_size: Number of sequences to generate.
        len_sequence: Length of each sequence.
        regime_change: If True, the noise distribution can change over time. Then a list of ditribution has to be defined.
        distribution_over_batch: How the distribution of the std is defined over the batch ('constant', 'uniform', 'log_uniform'). Default : 'constant
        multiple_configs: If True, obs_noise_distribution is a list of distributions and batch sizes.
        obs_noise_distribution: The noise distribution parameters :
            [distribution, values]  for a single regime (regime_change = False)
            [list of distributions, list of values, list of regime change instants] for multiple regimes (regime_change = True)
            [list of distributions, list of batch sizes] for multiple configurations (multiple_configs = True)
            A test function is implemented to verify shape coherence, suggesting fix in case of mismatch.
    """
    # Dummy obs_dim for shape inference
    obs_dim = 1

    # Call shape verification here
    obs_noise_distribution_shape_verification(
        regime_change, obs_noise_distribution, len_sequence, multiple_configs, batch_size, distribution_over_batch
    )

    def noise_distribution_for_R(
        size,
        len_seq,
        obs_dim,
        regime_change=False,
        obs_noise_distribution=None,
        distribution_over_batch="constant"
    ):
        # Use the same logic as noise_distribution, but with the above R
        if obs_noise_distribution is None:
            default_tensor = torch.tensor(1.0)
            default_value = float(default_tensor.sqrt().item())
            obs_noise_distribution = ['normal', default_value] if not regime_change else (['normal'], [default_value], [])

        obs_noise_distribution_shape_verification(regime_change, obs_noise_distribution, len_seq, False, None, distribution_over_batch)

        def sample_between(a, b, shape, log=False):
            a = torch.tensor(a, dtype=torch.float32)
            b = torch.tensor(b, dtype=torch.float32)
            if log:
                a = torch.log(a)
                b = torch.log(b)
                sample = torch.exp(torch.rand(*shape) * (b - a) + a)
            else:
                sample = torch.rand(*shape) * (b - a) + a
            return sample

        def single_regime_noise(size, len_seq, obs_dim, distribution, values, distribution_over_batch):
            def get_params_constant(values, param_shape):
                vals = torch.tensor(values, dtype=torch.float32)
                if vals.dim() == 0:
                    vals = vals.repeat(size, 1)
                elif vals.dim() == 1:
                    vals = vals.unsqueeze(0).repeat(size, 1)
                return vals

            def get_params_uniform(values, param_shape):
                minvals, maxvals = values
                minvals = torch.tensor(minvals, dtype=torch.float32)
                maxvals = torch.tensor(maxvals, dtype=torch.float32)
                if minvals.dim() == 0:
                    minvals = minvals.repeat(param_shape[1])
                if maxvals.dim() == 0:
                    maxvals = maxvals.repeat(param_shape[1])
                sampled = sample_between(minvals, maxvals, param_shape, log=False)
                return sampled

            def get_params_log_uniform(values, param_shape):
                minvals, maxvals = values
                minvals = torch.tensor(minvals, dtype=torch.float32)
                maxvals = torch.tensor(maxvals, dtype=torch.float32)
                if minvals.dim() == 0:
                    minvals = minvals.repeat(param_shape[1])
                if maxvals.dim() == 0:
                    maxvals = maxvals.repeat(param_shape[1])
                sampled = sample_between(minvals, maxvals, param_shape, log=True)
                return sampled

            if distribution_over_batch == "constant":
                get_params = get_params_constant
            elif distribution_over_batch == "uniform":
                get_params = get_params_uniform
            elif distribution_over_batch == "log_uniform":
                get_params = get_params_log_uniform
            else:
                raise ValueError(f"Unknown distribution_over_batch: {distribution_over_batch}")

            if distribution == 'normal':
                stds = get_params(values, (size, 1))
                stds = stds.view(size, 1, 1).expand(-1, len_seq, obs_dim)
                return stds
            elif distribution == 'bimodal':
                params = get_params(values, (size, 3))
                sigma1 = params[:, 0].view(-1, 1, 1)
                sigma2 = params[:, 1].view(-1, 1, 1)
                p1 = params[:, 2].view(-1, 1, 1)
                mask = torch.bernoulli(p1.expand(-1, len_seq, obs_dim))
                sigma1_exp = sigma1.expand(-1, len_seq, obs_dim)
                sigma2_exp = sigma2.expand(-1, len_seq, obs_dim)
                stds = torch.where(mask == 1.0, sigma1_exp, sigma2_exp)
                return stds
            else:
                raise ValueError("Unknown distribution type.")

        if not regime_change:
            distribution, values = obs_noise_distribution
            return single_regime_noise(size, len_seq, obs_dim, distribution, values, distribution_over_batch)
        else:
            distributions, values_list, transitions = obs_noise_distribution
            N = len(distributions)
            stds = torch.empty(size, len_seq, obs_dim)
            prev = 0
            for idx, t in enumerate(list(transitions) + [len_seq]):
                dist = distributions[idx]
                vals = values_list[idx]
                stds[:, prev:t, :] = single_regime_noise(size, t - prev, obs_dim, dist, vals, distribution_over_batch)
                prev = t
            return stds


    if not multiple_configs:
        obs_std_matrix = noise_distribution_for_R(
            batch_size, len_sequence, obs_dim,
            regime_change=regime_change,
            obs_noise_distribution=obs_noise_distribution,
            distribution_over_batch=distribution_over_batch
        )
    else:
        dist_list, batch_sizes = obs_noise_distribution
        all_obs_std = []
        for dist, batch_size in zip(dist_list, batch_sizes):
            obs_std = noise_distribution_for_R(
                batch_size, len_sequence, obs_dim,
                regime_change=regime_change,
                obs_noise_distribution=dist,
                distribution_over_batch=distribution_over_batch
            )
            all_obs_std.append(obs_std.squeeze(-1))
        obs_std_matrix = torch.cat(all_obs_std, dim=0)

    # Ensure output shape is (size, len_seq)
    if obs_std_matrix.dim() == 3 and obs_std_matrix.size(-1) == 1:
        obs_std_matrix = obs_std_matrix.squeeze(-1)

    return obs_std_matrix
