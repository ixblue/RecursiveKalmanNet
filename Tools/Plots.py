from matplotlib.axes import Axes
import torch
from matplotlib.colors import to_rgb


class CustomDefaultPlot(Axes):
    def __init__(self, fig, colors_list=None, markers_list=None, **kwargs):
        """
        Initialize custom Axes with lists of colors and markers, default grid settings and plot styles.
        Allows to have easy coherence in plots by specifying one index_format (int) per model in plot methods.
        """
        ax = fig.add_subplot(111, **kwargs)  
        super().__init__(fig, ax.get_position(), **kwargs)
        fig.delaxes(ax)  # Remove original subplot to prevent overlap
        fig.add_axes(self)  # Add the properly positioned custom Axes

        # Set default colors and markers or use user-provided lists
        self.colors_list = colors_list if colors_list else ['#1E88E5', '#D81B60', '#004D40', '#E65100', '#6A1B9A', '#FFB300']
        self.markers_list = markers_list if markers_list else ['d', 'v', '^', 'x', '*']

    def grid(self, *args, **kwargs):
        """
        Apply default grid formatting.
        """
        kwargs.setdefault('linewidth', 0.7)
        kwargs.setdefault('color', 'lightgray')
        kwargs.setdefault('alpha', 0.7)
        super().grid(*args, **kwargs)

    def plot(self, t, y, format_index=0, dark=False, **kwargs):
        """
        Overridden plot method to automatically apply default color and marker.
        """
        kwargs.setdefault('color', self.colors_list[format_index % len(self.colors_list)])
        kwargs.setdefault('marker', self.markers_list[format_index % len(self.markers_list)])

        if dark: # Darken the color if specified
            base_color = to_rgb(kwargs['color'])
            kwargs['color'] = tuple(0.6 * c for c in base_color)

        super().plot(t, y.detach().numpy(), **kwargs)
    
    def legend(self, **kwargs):
        """ Apply default legend formatting. """
        kwargs.setdefault('fontsize', 8)
        super().legend(**kwargs)


class StdComparisonPlot(CustomDefaultPlot):
    def __init__(self, fig, state=0, colors_list=None, markers_list=None, **kwargs):
        ax = fig.add_subplot(111, **kwargs)
        super().__init__(fig, colors_list=colors_list, markers_list=markers_list, **kwargs)
        fig.delaxes(ax)
        fig.add_axes(self)
        self.state = state

    def std_emp_plot(self, t, target, prediction, format_index=0, **kwargs):
        """
        Compute empirical standard deviation and plot with default color and marker.
        """
        errors = target[:, self.state, :] - prediction[:, self.state, :]
        std_devs = torch.sqrt(torch.mean(torch.square(errors - torch.mean(errors, dim=0)), dim=0))

        super().plot(t, std_devs, format_index=format_index, **kwargs)

    def std_pred_plot(self, t, cov, format_index=0, **kwargs):
        """
        Compute predicted standard deviation and plot with default color and marker.
        """
        if cov.dim() != 4:
            raise ValueError("Expected cov to have shape (batch_size, state_dim, state_dim, len_seq)")

        pred_std_devs = torch.sqrt(torch.mean(cov[:, self.state, self.state, :], dim=0))

        super().plot(t, pred_std_devs, format_index=format_index, **kwargs)

    def legend(self, **kwargs):
        """
        Override legend to apply default styling automatically.
        """
        kwargs.setdefault('loc', 'upper right')
        kwargs.setdefault('labelspacing', 0.10)
        kwargs.setdefault('ncol', 2)
        kwargs.setdefault('columnspacing', 0.2)
        kwargs.setdefault('handleheight', 2)
        kwargs.setdefault('handletextpad', 1)

        super().legend(**kwargs)


class ComparisonPlot(CustomDefaultPlot):
    def __init__(self, fig, colors_list=None, markers_list=None, **kwargs):
        ax = fig.add_subplot(111, **kwargs)
        super().__init__(fig, colors_list=colors_list, markers_list=markers_list, **kwargs)
        fig.delaxes(ax)
        fig.add_axes(self)



class LossCurvesPlot(CustomDefaultPlot):
    def __init__(self, fig, colors_list=None, markers_list=None, **kwargs):
        ax = fig.add_subplot(111, **kwargs)
        super().__init__(fig, colors_list=colors_list, markers_list=markers_list, **kwargs)
        fig.delaxes(ax)
        fig.add_axes(self)

    def plot_loss(self, model_name, format_index=0, **kwargs):
        """
        Plot training and validation loss curves from .results/loss_curves/{modelname}, using default styles.
        """
        path = f'.results/loss_curves/{model_name}.pt'
        losses = torch.load(path)

        t = torch.arange(len(losses['train_losses']))
        train_losses = losses['train_losses']
        val_losses = losses['cv_losses']

        # Training curve (dotted)
        train_kwargs = kwargs.copy()
        train_kwargs.setdefault('label', f'{model_name} Training')
        train_kwargs.setdefault('linestyle', '--')
        train_kwargs.pop('marker', None)
        super().plot(t, train_losses, format_index=format_index, **train_kwargs)

        # Validation curve (solid)
        val_kwargs = kwargs.copy()
        val_kwargs.setdefault('label', f'{model_name} Validation')
        val_kwargs.setdefault('linestyle', '-')
        val_kwargs.pop('marker', None)
        super().plot(t, val_losses, format_index=format_index, dark=True, **val_kwargs)
