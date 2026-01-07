from pathlib import Path
import math
import string

import matplotlib.pyplot as plt
from PIL import Image

final_panel_figures = {
    "local_analysis_tricontour": {
        "Title": None,
        "Paths": [
            "final_figures/tricontour_app_vs_pos_signal_local_analysis.png",
            "final_figures/tricontour_app_vs_pos_signal_gamma.png",
            "final_figures/tricontour_app_vs_pos_signal_no_quartile.png",
            "final_figures/tricontour_app_vs_pos_signal_no_quartile_gamma.png",
        ],
        "subtitles": [
            "Local Analysis",
            "Randomized Max Applications (Gamma)",
            "Randomized Application Pattern (No Quartile)",
            "Randomized Max Applications (Gamma) and No Quartile",
        ],
        "output_path": "final_figures/local_analysis_4_panel_tricontour.png",
        # Optional per-figure overrides go here, e.g.:
        # "settings": {"figsize": (7.5, 7.5), "subplot_title_fontsize": 7},
    },
    "local_analysis_binned": {
        "Title": None,
        "Paths": [
            "final_figures/binned_app_vs_pos_signal_local_analysis.png",
            "final_figures/binned_app_vs_pos_signal_gamma.png",
            "final_figures/binned_app_vs_pos_signal_no_quartile.png",
            "final_figures/binned_app_vs_pos_signal_no_quartile_gamma.png",
        ],
        "subtitles": [
            "Local Analysis",
            "Randomized Max Applications (Gamma)",
            "Randomized Application Pattern (No Quartile)",
            "Randomized Max Applications (Gamma) and No Quartile",
        ],
        "output_path": "final_figures/local_analysis_4_panel_binned.png",
    },
    "base_case": {
        "Title": None,
        "Paths": [
            "final_figures/tricontour_app_vs_pos_signal_base_case.png",
            "final_figures/binned_app_vs_pos_signal_base_case.png",
        ],
        "subtitles": [
            "Global Analysis Tri-Contour",
            "Global Analysis Binned Optimal Signal",
        ],
        "output_path": "final_figures/global_analysis_2_panel.png",
        # Make this one vertical
        "settings": {
            "two_panel_orientation": "horizontal", 
            "figsize": (7.5, 15.0),
            "suptitle_fontsize": 10},
        
    },
}

FIGURE_SETTINGS = dict(
    # Layout:
    # - If layout is None, it is inferred from the number of images.
    # - For 2-panel figures, choose "horizontal" (1x2) or "vertical" (2x1).
    layout=None,
    two_panel_orientation="vertical",  # "horizontal" or "vertical"

    # Figure appearance
    figsize=(7.5, 7.5),
    dpi=600,

    # Titles
    suptitle_fontsize=14,
    suptitle_y=0.99,                 # move figure title up/down
    subplot_title_fontsize=8,        # smaller to avoid overlap
    subplot_title_pad=2,

    # Panel labels (A/B/C/...)
    panel_labels="auto",             # "auto" or a tuple/list like ("A", "B", ...)
    label_xy=(0.02, 0.98),
    label_kwargs=dict(
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8),
    ),

    # Axes display options
    show_axes=False,
    hide_ticks=True,

    # Subplot spacing
    wspace=0.03,
    hspace=0.04,

    # Margins (in figure fraction coordinates). Increase top to reduce gap below suptitle.
    # Decrease top to add more space between suptitle and subplots.
    margins=dict(left=0.01, right=0.99, bottom=0.01, top=0.94),

    # Saving: bbox_inches="tight" trims whitespace; pad_inches adds whitespace back.
    save_bbox_inches="tight",        # "tight" or None
    save_pad_inches=0.12,
)

# Default axis labels (can be overridden per figure below)
DEFAULT_XLABEL = "Applicants"
DEFAULT_YLABEL = "Positions"

def _infer_layout(n_panels: int, two_panel_orientation: str) -> tuple[int, int]:
    if n_panels <= 0:
        raise ValueError("n_panels must be >= 1")

    if n_panels == 1:
        return (1, 1)
    if n_panels == 2:
        if two_panel_orientation.lower() not in {"horizontal", "vertical"}:
            raise ValueError('two_panel_orientation must be "horizontal" or "vertical"')
        return (1, 2) if two_panel_orientation.lower() == "horizontal" else (2, 1)
    if n_panels == 4:
        return (2, 2)

    # Generic fallback: near-square grid
    cols = math.ceil(math.sqrt(n_panels))
    rows = math.ceil(n_panels / cols)
    return (rows, cols)


def panel_png(
    paths,
    out_path="panel.png",
    *,
    layout=None,
    two_panel_orientation="horizontal",
    figsize=(7, 7),
    dpi=600,
    panel_labels="auto",
    label_xy=(0.02, 0.98),
    label_kwargs=None,
    suptitle=None,
    suptitle_fontsize=14,
    suptitle_y=0.99,
    titles=None,
    subplot_title_fontsize=9,
    subplot_title_pad=2,
    xlabels=None,
    ylabels=None,
    global_xlabel=None,
    global_ylabel=None,
    show_axes=False,
    hide_ticks=True,
    wspace=0.02,
    hspace=0.02,
    margins=None,
    save_bbox_inches="tight",
    save_pad_inches=0.10,
):
    paths = [Path(p) for p in paths]
    n_imgs = len(paths)

    if layout is None:
        layout = _infer_layout(n_imgs, two_panel_orientation)

    n_axes = layout[0] * layout[1]
    if n_imgs > n_axes:
        raise ValueError(f"Layout {layout} only has {n_axes} slots but you provided {n_imgs} images.")
    if n_imgs < n_axes:
        # We'll create the grid and hide unused axes.
        pass

    # Panel labels
    if panel_labels == "auto":
        panel_labels = tuple(string.ascii_uppercase[:n_imgs])
    elif panel_labels is None:
        panel_labels = None
    else:
        if len(panel_labels) != n_imgs:
            raise ValueError(f"panel_labels must have length {n_imgs} (or use 'auto').")

    if label_kwargs is None:
        label_kwargs = dict(
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8),
        )

    # Normalize titles/xlabels/ylabels to per-panel lists (or None)
    def _normalize(val, name, n):
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            if len(val) != n:
                raise ValueError(f"{name} must have length {n}.")
            return list(val)
        return [val] * n

    titles = _normalize(titles, "titles", n_imgs)
    xlabels = _normalize(xlabels, "xlabels", n_imgs)
    ylabels = _normalize(ylabels, "ylabels", n_imgs)

    imgs = [Image.open(p).convert("RGBA") for p in paths]

    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax in axes_flat[n_imgs:]:
        ax.set_axis_off()

    for i, (ax, im) in enumerate(zip(axes_flat[:n_imgs], imgs)):
        ax.imshow(im)

        if show_axes:
            ax.set_axis_on()
            if hide_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.set_axis_off()

        # Panel labels (A, B, C, ...)
        if panel_labels:
            ax.text(label_xy[0], label_xy[1], panel_labels[i],
                    transform=ax.transAxes, **label_kwargs)

        # Per-panel title + axis labels
        if titles is not None:
            ax.set_title(
                titles[i],
                fontsize=subplot_title_fontsize,
                pad=subplot_title_pad,
                wrap=True,
            )
        if xlabels is not None:
            ax.set_xlabel(xlabels[i])
        if ylabels is not None:
            ax.set_ylabel(ylabels[i])

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=suptitle_y)

    if global_xlabel:
        try:
            fig.supxlabel(global_xlabel)
        except AttributeError:
            fig.text(0.5, 0.01, global_xlabel, ha="center", va="bottom")
    if global_ylabel:
        try:
            fig.supylabel(global_ylabel)
        except AttributeError:
            fig.text(0.01, 0.5, global_ylabel, ha="left", va="center", rotation=90)

    if margins is None:
        margins = dict(left=0.01, right=0.99, bottom=0.01, top=0.94)

    fig.subplots_adjust(wspace=wspace, hspace=hspace, **margins)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches=save_bbox_inches, pad_inches=save_pad_inches)
    plt.close(fig)


if __name__ == "__main__":
    for analysis, info in final_panel_figures.items():
        per_figure_settings = dict(FIGURE_SETTINGS)
        per_figure_settings.update(info.get("settings", {}))

        panel_png(
            paths=info["Paths"],
            out_path=info["output_path"],
            suptitle=info["Title"],
            titles=info["subtitles"],
            xlabels=DEFAULT_XLABEL,
            ylabels=DEFAULT_YLABEL,
            **per_figure_settings,
        )
