"""Kleurenpalet: één bron van waarheid voor alle grafiekkleuren.

De kleuren zijn gevalideerd met de dataviz-paletvalidator: kleurenblind-veilig
(voldoende CVD-afstand tussen aangrenzende kleuren), genoeg chroma om niet als
grijs te lezen en voldoende contrast op de achtergrond. Licht en donker zijn
apárt gekozen en gevalideerd — de donkere variant is geen automatische flip
maar andere stappen van dezelfde tinten, afgestemd op een donker oppervlak.

Deze module importeert bewust geen streamlit of plotly: de app bepaalt zelf de
actieve modus (via het Streamlit-thema) en vraagt hier alleen kleuren op.
"""

# Sequentiële ramp (één tint blauw, licht -> donker) voor continue waarden
# zoals hartslag op een scatter of een heatmap. Licht = laag, donker = hoog.
SEQ_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b",
]

# Per modus: sportkleuren, zonekleuren (Z1..Z5, koel -> heet), inkt- en
# hulpkleuren. "ink" is voor nadruk (bijv. de LTHR-lijn), "muted" voor
# assen/bijschriften, "ref_line" voor richtwaarde-lijnen, "surface" is het
# grafiekoppervlak waartegen het palet gevalideerd is.
_LICHT = {
    "sport": {"Hardlopen": "#e34948", "Fietsen": "#2a78d6", "Zwemmen": "#1baf7a"},
    "zones": ["#1c5cab", "#008300", "#eda100", "#eb6834", "#b93534"],
    # Algemene categorische reeks (vaste volgorde, nooit cyclen) voor
    # categorieën zónder eigen betekeniskleur, zoals zwemslagen.
    "cats": ["#2a78d6", "#1baf7a", "#eda100", "#008300",
             "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    "seq": SEQ_BLUE,
    "ink": "#0b0b0b",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "ref_line": "#52514e",
    "surface": "#fcfcfb",
}

_DONKER = {
    "sport": {"Hardlopen": "#e66767", "Fietsen": "#3987e5", "Zwemmen": "#199e70"},
    "zones": ["#3987e5", "#008300", "#c98500", "#d95926", "#e66767"],
    "cats": ["#3987e5", "#199e70", "#c98500", "#008300",
             "#9085e9", "#e66767", "#d55181", "#d95926"],
    "seq": SEQ_BLUE,
    "ink": "#ffffff",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "ref_line": "#c3c2b7",
    "surface": "#1a1a19",
}


def get_palette(mode: str | None) -> dict:
    """Palet voor de gevraagde modus; alles behalve "dark" geeft licht."""
    return _DONKER if mode == "dark" else _LICHT


def with_alpha(hex_color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)' voor transparante vlakken en banden."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
