"""Grafiekhelpers: gedeelde opmaak en gedrag voor alle Plotly-figuren.

Pure plotly/pandas-helpers, bewust zonder streamlit-import: de app rendert de
figuren, deze module zorgt dat ze er allemaal hetzelfde uitzien en zich
hetzelfde gedragen (assen, legenda, enkelpunts-zoom, modebar).
"""

import pandas as pd

# Modebar-instellingen voor elke grafiek: geen plotly-logo en geen knoppen
# die op een dashboard alleen maar per ongeluk aangeklikt worden.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def style_fig(fig, show_legend: bool = True):
    """Huisstijl voor alle grafieken: legenda horizontaal ónder de grafiek."""
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0, title=None),
        showlegend=show_legend,
        margin=dict(t=30, b=10, l=10, r=10),
    )
    return fig


def date_xaxis(fig, dates):
    """Nette datum-as: alleen dd-mm-jjjj, geen uur-ticks binnen de dag.

    Bij weinig metingen zetten we de ticks precies op de meetdagen; bij veel
    metingen laten we Plotly zelf nette dag/week/maand-labels kiezen.
    """
    uniek = pd.to_datetime(pd.Series(list(dates))).dt.normalize().drop_duplicates()
    fig.update_xaxes(tickformat="%d-%m-%Y")
    if 1 <= len(uniek) <= 15:
        fig.update_xaxes(tickmode="array", tickvals=sorted(uniek))
    return fig


def pace_as_time(seconds: pd.Series) -> pd.Series:
    """Tempo in seconden -> datetime, zodat de as nette M:SS-labels krijgt."""
    return pd.to_datetime(seconds, unit="s")


def pace_axis(fig):
    """Tempo-as: M:SS-labels en omgekeerd (sneller = hoger in de grafiek)."""
    fig.update_yaxes(tickformat="%M:%S", autorange="reversed")
    return fig


def pad_single_point(fig, x_values, days: int = 4,
                     y_center=None, y_pad=None, reversed_y: bool = False):
    """Voorkom extreem inzoomen als een grafiek maar één (x-)meetpunt heeft.

    Zet dan een vast venster van ``±days`` rond het punt en — als ``y_center``
    en ``y_pad`` zijn meegegeven — ook een vast y-bereik. Voor een omgekeerde
    tempo-as (``reversed_y=True``) wordt het y-bereik omgedraaid, omdat een
    expliciet bereik de eerdere ``autorange="reversed"`` vervangt.
    """
    uniek = pd.to_datetime(pd.Series(list(x_values))).drop_duplicates()
    if len(uniek) != 1:
        return fig
    t = pd.Timestamp(uniek.iloc[0])
    fig.update_xaxes(range=[t - pd.Timedelta(days=days), t + pd.Timedelta(days=days)],
                     tickformat="%d-%m-%Y")
    if y_center is not None and y_pad is not None:
        laag, hoog = y_center - y_pad, y_center + y_pad
        fig.update_yaxes(range=[hoog, laag] if reversed_y else [laag, hoog])
    return fig


def add_race_marker(fig, race_date, name: str, color: str):
    """Verticale stippellijn + label op de racedatum, als die in beeld is."""
    fig.add_vline(x=pd.Timestamp(race_date), line_dash="dot", line_color=color,
                  annotation_text=name, annotation_position="top left",
                  annotation_font_color=color)
    return fig
