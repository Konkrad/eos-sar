import io

import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable


def fig2img(fig, **kwargs):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    img = imageio.imread(buf)
    return img


class Bounds:
    """A fixed `(min, max)` value range, used to normalize an array for display."""

    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def get_bounds(self):
        """Return the `(minval, maxval)` bounds."""
        return self.minval, self.maxval


class PercentileBounds(Bounds):
    """`Bounds` computed from the low/high percentiles of an array's values."""

    def __init__(self, array, percentile=5):
        """
        Parameters
        ----------
        array : ndarray
            Array to compute percentiles from (nans are ignored).
        percentile : float, optional
            Percentile used for the lower bound; the upper bound uses
            `100 - percentile`. Must be in `[0, 50]`; out-of-range values
            fall back to 5 (with a warning printed). Defaults to 5.
        """
        if percentile < 0:
            print("Warning: Negative percentile given, defaulting to 5")
            percentile = 5
        elif percentile > 50:
            print("Warning: Percentile greater then 50, defaulting to 5")
            percentile = 5
        minval = np.nanpercentile(array, percentile)
        maxval = np.nanpercentile(array, 100 - percentile)
        super().__init__(minval, maxval)


def to_uint8(array, bounds_provider):
    """
    Linearly rescale `array` to uint8 `[0, 255]`, clipping to `bounds_provider`'s bounds.

    Parameters
    ----------
    array : ndarray
        Array to rescale.
    bounds_provider : Bounds
        Provides the `(minval, maxval)` range mapped to `[0, 255]`.

    Returns
    -------
    ndarray of uint8
        Rescaled and clipped array, same shape as `array`.
    """
    minval, maxval = bounds_provider.get_bounds()
    # linear function to map minval to 0 and maxval to 255
    normalized = (array - minval) / (maxval - minval) * 255

    normalized[normalized < 0] = 0
    normalized[normalized > 255] = 255
    return normalized.astype(np.uint8)


def phase_to_jet(image):
    """
    Colorize a wrapped phase array (`[-pi, pi]`) with the "jet" colormap.

    Parameters
    ----------
    image : ndarray
        Phase array, values expected in `[-pi, pi]`.

    Returns
    -------
    ndarray of uint8
        RGB image, shape `image.shape + (3,)`.
    """
    # Get the color map by name:
    cm = plt.get_cmap("jet")

    # Apply the colormap like a function to any array:
    colored_image = cm(to_uint8(image, Bounds(-np.pi, np.pi)))

    # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
    # But we want to convert to RGB in uint8 and save it
    return (colored_image[:, :, :3] * 255).astype(np.uint8)


def sar_amp_to_pretty_uint8(amp, p=5):
    """Rescale a SAR amplitude array to uint8 using `p`/`100-p` percentile bounds (see `PercentileBounds`)."""
    return to_uint8(amp, PercentileBounds(amp, p))


def save_imgs_as_gif(gif_path, images, duration=0.1):
    """Save a sequence of `images` as an animated GIF at `gif_path`."""
    imageio.mimsave(gif_path, images, duration=duration)


def plot_phi(
    phi,
    cmap="jet",
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    """
    Display (and optionally save) a wrapped phase image with a colorbar.

    Parameters
    ----------
    phi : ndarray
        Wrapped phase array, displayed with `vmin=-pi`, `vmax=pi`.
    cmap : str, optional
        Matplotlib colormap name. Defaults to "jet".
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size passed to `plt.subplots`.
    remove_ticks : bool, optional
        If True (default), hide the axis ticks.
    fig_out_path : str, optional
        If given, also save the figure to this path.
    **save_fig_kwargs
        Extra keyword arguments passed to `fig.savefig`.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(phi, cmap=cmap, interpolation="nearest", vmin=-np.pi, vmax=np.pi)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, orientation="vertical", label="Phase (rad)")

    ax.set_title(title)

    if remove_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()


def plot_amp(
    amp,
    vmin,
    vmax,
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    """
    Display (and optionally save) a SAR amplitude image with a colorbar.

    Parameters
    ----------
    amp : ndarray
        Amplitude array, displayed in grayscale.
    vmin, vmax : float
        Display bounds for `amp`.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size passed to `plt.subplots`.
    remove_ticks : bool, optional
        If True (default), hide the axis ticks.
    fig_out_path : str, optional
        If given, also save the figure to this path.
    **save_fig_kwargs
        Extra keyword arguments passed to `fig.savefig`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(amp, cmap="gray", vmin=vmin, vmax=vmax)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax, orientation="vertical", label="Amplitude")

    ax.set_title(title)

    if remove_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()


def plot_amp_phi(
    amp,
    vmin,
    vmax,
    phi,
    phi_cmap="jet",
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    """
    Display (and optionally save) amplitude and wrapped phase side by side.

    Parameters
    ----------
    amp : ndarray
        Amplitude array, displayed in grayscale.
    vmin, vmax : float
        Display bounds for `amp`.
    phi : ndarray
        Wrapped phase array, displayed with `vmin=-pi`, `vmax=pi`.
    phi_cmap : str, optional
        Matplotlib colormap name for `phi`. Defaults to "jet".
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size passed to `plt.subplots`.
    remove_ticks : bool, optional
        If True (default), hide the axis ticks.
    fig_out_path : str, optional
        If given, also save the figure to this path.
    **save_fig_kwargs
        Extra keyword arguments passed to `fig.savefig`.
    """
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    im_amp = axs[0].imshow(amp, cmap="gray", vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(axs[0])
    cax_amp = divider.append_axes("bottom", size="5%", pad=0.3)
    fig.colorbar(im_amp, cax=cax_amp, orientation="horizontal", label="Amplitude")

    im = axs[1].imshow(
        phi, cmap=phi_cmap, interpolation="nearest", vmin=-np.pi, vmax=np.pi
    )

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Phase (rad)")

    if remove_ticks:
        for ax in axs:
            ax.set_yticks([])
            ax.set_xticks([])

    fig.suptitle(title)

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()


def cmpx_interf_to_rgb(amp, vmin, vmax, phi):
    """
    Encode a complex interferogram as an HSV-based RGB image (hue=phase, value=amplitude).

    Parameters
    ----------
    amp : ndarray
        Amplitude array, clipped to `[vmin, vmax]` and mapped to the HSV
        "value" channel.
    vmin, vmax : float
        Display bounds for `amp`.
    phi : ndarray
        Wrapped phase array (`[-pi, pi]`), mapped to the HSV "hue" channel.

    Returns
    -------
    ndarray
        RGB image in `[0, 1]`, shape `amp.shape + (3,)`.
    """
    # use angle to determine hue, normalized from 0-1
    min_phi = -np.pi
    max_phi = np.pi
    h = (phi - min_phi) / (max_phi - min_phi)

    # value is set as a function of amplitude, normalized
    amp_clipped = np.clip(amp, vmin, vmax)
    v = (amp_clipped - vmin) / (vmax - vmin)

    # saturation taken as 1
    s = np.ones_like(v)

    hsv = np.stack([h, s, v], axis=2)

    c = hsv_to_rgb(hsv)

    return c


def plot_cmpx_interf_rgb(
    amp,
    vmin,
    vmax,
    phi,
    title="",
    figsize=None,
    remove_ticks=True,
    fig_out_path=None,
    **save_fig_kwargs,
):
    """
    Display (and optionally save) a complex interferogram as an HSV-encoded RGB image.

    Combines `cmpx_interf_to_rgb` with a phase colorbar (hue).

    Parameters
    ----------
    amp : ndarray
        Amplitude array, mapped to the HSV "value" channel.
    vmin, vmax : float
        Display bounds for `amp`.
    phi : ndarray
        Wrapped phase array (`[-pi, pi]`), mapped to the HSV "hue" channel.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size passed to `plt.subplots`.
    remove_ticks : bool, optional
        If True (default), hide the axis ticks.
    fig_out_path : str, optional
        If given, also save the figure to this path.
    **save_fig_kwargs
        Extra keyword arguments passed to `fig.savefig`.
    """
    c = cmpx_interf_to_rgb(amp, vmin, vmax, phi)

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(c)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    mappable = ScalarMappable(Normalize(vmin=-np.pi, vmax=np.pi), cmap="hsv")

    fig.colorbar(mappable, cax=cax, orientation="vertical", label="Phase(rad)")

    ax.set_title(title)

    if remove_ticks:
        ax.set_yticks([])
        ax.set_xticks([])

    fig.tight_layout()

    if fig_out_path is not None:
        fig.savefig(fig_out_path, **save_fig_kwargs)

    plt.show()
