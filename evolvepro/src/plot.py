"""
plot.py — Plotting utilities for EvolvePro.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _count_mutations(variant: str) -> int:
    """Return the number of mutations encoded in a variant string.

    Mutations are assumed to be separated by underscores
    (e.g. ``"D23M_A107L_F190R"`` → 3).  A single-mutation variant
    with no underscore (e.g. ``"F190R"``) returns 1.

    Args:
        variant: Variant name string.

    Returns:
        Integer count of mutations.
    """
    return len(variant.split("_"))


def plot_y_pred_distribution(
    df: pd.DataFrame,
    *,
    bin_width: float = 0.05,
    by_n_mutations: bool = False,
    highlight_wt: bool = False,
    wt_label: str = "WT",
    wt_color: str = "gold",
    wt_alpha: float = 0.35,
    variant_col: str = "variant",
    y_pred_col: str = "y_pred",
    palette: str = "tab10",
    stat: str = "count",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Plot the distribution of predicted fitness scores (``y_pred``).

    Parameters
    ----------
    df:
        DataFrame containing at least a ``y_pred`` column and, when
        ``by_n_mutations=True``, a ``variant`` column from which mutation
        counts are derived.
    bin_width:
        Width of each histogram bin along the ``y_pred`` axis.  Smaller
        values produce finer-grained bins.  Defaults to ``0.05``.
    by_n_mutations:
        When ``True``, overlay a separate histogram for each distinct
        mutation count found in the data, coloured by the ``palette``.
        When ``False`` (default), a single histogram is drawn for the
        whole dataset.
    highlight_wt:
        When ``True``, shade the bin that the wild-type (WT) variant
        falls in and draw a vertical dashed line at its exact ``y_pred``
        value.  The WT row is identified by matching ``variant_col``
        against ``wt_label``.  A warning is printed (and the flag is
        silently ignored) if no matching row is found.
    wt_label:
        The variant name used to locate the WT row.  Defaults to
        ``"WT"``.
    wt_color:
        Colour of the WT bin highlight and dashed line.  Any Matplotlib
        colour string is accepted.  Defaults to ``"gold"``.
    variant_col:
        Name of the column that holds variant identifiers used to count
        mutations and locate the WT row.
    y_pred_col:
        Name of the column containing predicted fitness values.
        Defaults to ``"y_pred"``.
    palette:
        Any Seaborn / Matplotlib named colour palette used when
        ``by_n_mutations=True``.  Defaults to ``"tab10"``.
    stat:
        Aggregate statistic for the y-axis — any value accepted by
        :func:`seaborn.histplot` (``"count"``, ``"frequency"``,
        ``"density"``, ``"probability"``).  Defaults to ``"count"``.
    ax:
        Optional existing :class:`matplotlib.axes.Axes` to draw on.
        When ``None`` a new figure is created.
    figsize:
        ``(width, height)`` in inches for the newly created figure.
        Ignored when *ax* is provided.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.

    Examples
    --------
    >>> import pandas as pd
    >>> from plot import plot_y_pred_distribution
    >>> df = pd.read_csv("df_sorted_all.csv")

    # Single histogram, default 0.05-wide bins
    >>> fig = plot_y_pred_distribution(df)

    # Overlaid by mutation count, coarser bins
    >>> fig = plot_y_pred_distribution(df, bin_width=0.1, by_n_mutations=True)
    """
    if y_pred_col not in df.columns:
        raise ValueError(
            f"Column '{y_pred_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    if by_n_mutations and variant_col not in df.columns:
        raise ValueError(
            f"Column '{variant_col}' not found in DataFrame, which is required "
            f"when by_n_mutations=True. Available columns: {df.columns.tolist()}"
        )

    plot_df = df.copy()

    if by_n_mutations:
        plot_df["n_mutations"] = plot_df[variant_col].apply(_count_mutations)
        # Sort so legend is ordered 1, 2, 3, …
        sorted_counts = sorted(plot_df["n_mutations"].unique())
        plot_df["n_mutations"] = pd.Categorical(
            plot_df["n_mutations"], categories=sorted_counts, ordered=True
        )
        hue_col: str | None = "n_mutations"
        hue_label = "# mutations"
    else:
        hue_col = None
        hue_label = None

    y_min = plot_df[y_pred_col].min()
    y_max = plot_df[y_pred_col].max()
    bins = list(
        _frange(y_min - bin_width, y_max + 2 * bin_width, bin_width)
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.histplot(
        data=plot_df,
        x=y_pred_col,
        hue=hue_col,
        bins=bins,
        stat=stat,
        palette=palette if hue_col else None,
        element="bars",
        multiple="layer",
        alpha=0.55 if hue_col else 0.75,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )

    # ------------------------------------------------------------------
    # WT bin highlight
    # ------------------------------------------------------------------
    if highlight_wt:
        wt_rows = plot_df[plot_df[variant_col] == wt_label]
        if wt_rows.empty:
            import warnings
            warnings.warn(
                f"highlight_wt=True but no row with {variant_col}=='{wt_label}' "
                f"was found. Skipping WT highlight.",
                UserWarning,
                stacklevel=2,
            )
        else:
            wt_y_pred = wt_rows[y_pred_col].iloc[0]

            # Vertical dashed line at the exact WT value
            ax.axvline(
                wt_y_pred,
                color=wt_color,
                linewidth=1.5,
                linestyle="--",
                zorder=3,
            )

            # Grab handles/labels from the seaborn-generated legend (hue),
            # then append the WT line so both appear in one legend.
            existing_legend = ax.get_legend()
            if existing_legend is not None:
                existing_handles = existing_legend.legend_handles
                existing_labels = [t.get_text() for t in existing_legend.get_texts()]
            else:
                existing_handles, existing_labels = [], []
            wt_handle = plt.Line2D([0], [0], color=wt_color, linewidth=1.5, linestyle="--")
            ax.legend(
                handles=existing_handles + [wt_handle],
                labels=existing_labels + [f"{wt_label} y_pred = {wt_y_pred:.3f}"],
                title=hue_label if hue_col else None,
                loc="best",
                framealpha=0.8,
            )

    ax.set_xlabel("Predicted fitness (y_pred)", fontsize=12)
    ax.set_ylabel(stat.capitalize(), fontsize=12)
    ax.set_title("Distribution of predicted fitness scores", fontsize=13)

    if hue_col and not highlight_wt:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(hue_label)

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_additive_vs_actual(
    df: pd.DataFrame,
    *,
    variant_col: str = "variant",
    y_pred_col: str = "y_pred",
    by_n_mutations: bool = False,
    color: str = "steelblue",
    palette: str = "tab10",
    alpha: float = 0.6,
    point_size: float = 30,
    show_diagonal: bool = True,
    diagonal_color: str = "crimson",
    diagonal_linestyle: str = "--",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 6),
) -> plt.Figure:
    """Scatter plot of additive predicted fitness vs. actual predicted fitness
    for all multi-mutant variants.

    For each multi-mutant variant the **additive score** is calculated as the
    sum of the ``y_pred`` values of each constituent single mutation.  Only
    variants whose individual component mutations all appear as single-mutant
    rows in *df* are included; multi-mutants with a missing component are
    silently skipped.

    Parameters
    ----------
    df:
        DataFrame containing at least a ``variant`` column and a ``y_pred``
        column.  Single-mutant rows are identified by the absence of an
        underscore in the variant name.
    variant_col:
        Name of the column holding variant identifiers.  Defaults to
        ``"variant"``.
    y_pred_col:
        Name of the column containing predicted fitness values.  Defaults to
        ``"y_pred"``.
    by_n_mutations:
        When ``True``, points are coloured by the number of mutations in the
        variant using the ``palette``.  A legend entry is added for each
        distinct mutation count.  When ``False`` (default), all points share
        the single ``color``.
    color:
        Colour used for all scatter points when ``by_n_mutations=False``.
        Defaults to ``"steelblue"``.
    palette:
        Any Seaborn / Matplotlib named colour palette used to assign colours
        per mutation count when ``by_n_mutations=True``.  Defaults to
        ``"tab10"``.
    alpha:
        Opacity of the scatter points.  Defaults to ``0.6``.
    point_size:
        Marker size passed to :func:`matplotlib.axes.Axes.scatter`.  Defaults
        to ``30``.
    show_diagonal:
        When ``True`` (default), draw a diagonal ``y = x`` reference line so
        that epistatic deviations from additivity are immediately visible.
    diagonal_color:
        Colour of the ``y = x`` reference line.  Defaults to ``"crimson"``.
    diagonal_linestyle:
        Line style of the ``y = x`` reference line.  Defaults to ``"--"``.
    ax:
        Optional existing :class:`matplotlib.axes.Axes` to draw on.  When
        ``None`` a new figure is created.
    figsize:
        ``(width, height)`` in inches for the newly created figure.  Ignored
        when *ax* is provided.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the scatter plot.

    Raises
    ------
    ValueError
        If ``variant_col`` or ``y_pred_col`` are not present in *df*.

    Examples
    --------
    >>> import pandas as pd
    >>> from plot import plot_additive_vs_actual
    >>> df = pd.read_csv("df_sorted_all.csv")

    # Single colour
    >>> fig = plot_additive_vs_actual(df)

    # Colour points by number of mutations
    >>> fig = plot_additive_vs_actual(df, by_n_mutations=True)
    >>> fig.savefig("additive_vs_actual.png", dpi=150)
    """
    if variant_col not in df.columns:
        raise ValueError(
            f"Column '{variant_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )
    if y_pred_col not in df.columns:
        raise ValueError(
            f"Column '{y_pred_col}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Build a lookup table of single-mutant y_pred values.
    single_mask = df[variant_col].apply(lambda v: "_" not in str(v))
    single_lookup: dict[str, float] = (
        df.loc[single_mask].set_index(variant_col)[y_pred_col].to_dict()
    )

    additive_scores: list[float] = []
    actual_scores: list[float] = []
    n_mutations_list: list[int] = []

    for _, row in df[~single_mask].iterrows():
        components = str(row[variant_col]).split("_")
        # Skip if any component single mutant is absent from the dataset.
        if not all(c in single_lookup for c in components):
            continue
        additive_scores.append(sum(single_lookup[c] for c in components))
        actual_scores.append(row[y_pred_col])
        n_mutations_list.append(len(components))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    legend_handles: list = []

    if by_n_mutations:
        sorted_counts = sorted(set(n_mutations_list))
        colors = sns.color_palette(palette, n_colors=len(sorted_counts))
        color_map = dict(zip(sorted_counts, colors))

        for n in sorted_counts:
            idx = [i for i, v in enumerate(n_mutations_list) if v == n]
            x_vals = [additive_scores[i] for i in idx]
            y_vals = [actual_scores[i] for i in idx]
            handle = ax.scatter(
                x_vals,
                y_vals,
                c=[color_map[n]],
                alpha=alpha,
                s=point_size,
                linewidths=0,
                label=f"{n} mutations",
                zorder=3,
            )
            legend_handles.append(handle)
    else:
        ax.scatter(
            additive_scores,
            actual_scores,
            c=color,
            alpha=alpha,
            s=point_size,
            linewidths=0,
            zorder=3,
        )

    if show_diagonal:
        all_vals = additive_scores + actual_scores
        lo, hi = min(all_vals), max(all_vals)
        diag_line, = ax.plot(
            [lo, hi],
            [lo, hi],
            color=diagonal_color,
            linestyle=diagonal_linestyle,
            linewidth=1.2,
            label="y = x (additive)",
            zorder=2,
        )
        legend_handles.append(diag_line)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="# mutations" if by_n_mutations else None,
            framealpha=0.8,
            loc="best",
        )

    ax.set_xlabel("Additive score (sum of single-mutant y_pred)", fontsize=12)
    ax.set_ylabel("Predicted fitness (y_pred)", fontsize=12)
    ax.set_title("Additive vs. predicted fitness for multi-mutant variants", fontsize=13)

    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _frange(start: float, stop: float, step: float):
    """Yield evenly spaced floats from *start* up to (not including) *stop*."""
    x = start
    while x < stop:
        yield round(x, 10)
        x += step