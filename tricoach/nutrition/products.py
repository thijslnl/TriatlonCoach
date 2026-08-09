"""De productdatabase: wat zit er in de gels, dranken en repen die ik gebruik.

Eén rij per product in de SQLite-tabel ``nutrition_products``, bewerkbaar op de
Voeding-tab. Bij een lege tabel wordt hij gevuld met :data:`DEFAULT_PRODUCTS` —
de producten die daadwerkelijk in de kast liggen. Die startwaarden komen van de
verpakkingen en zijn **indicatief**: fabrikanten wijzigen recepturen, dus alles
is aanpasbaar en de UI zegt dat er ook bij.

De belangrijkste kolom is ``source``: de koolhydraatbron bepaalt het
opnameplafond (zie :mod:`tricoach.nutrition.rules`). Er zijn drie waarden:

- ``single`` — alleen glucose/maltodextrine: plafond ~60 g/uur.
- ``dual`` — glucose + fructose (ratio ~1:0,8): plafond ~90 g/uur.
- ``onbekend`` — de verpakking noemt meerdere suikers maar geen verhouding.
  Die telt in de planner **conservatief als single-source**: een plafond dat te
  laag blijkt kost je wat prestatie, een plafond dat te hoog blijkt kost je je
  race.
"""

import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

# Producttypes. Het type bepaalt hoe de planner het product inzet: drank levert
# koolhydraten én vocht (en valt onder de concentratielimiet), gels en vast
# voedsel leveren alleen koolhydraten.
KIND_GEL, KIND_DRINK, KIND_SOLID = "gel", "drank", "vast"
KINDS = (KIND_GEL, KIND_DRINK, KIND_SOLID)

# Koolhydraatbronnen; zie de moduledocstring.
SOURCE_SINGLE, SOURCE_DUAL, SOURCE_UNKNOWN = "single", "dual", "onbekend"
SOURCES = (SOURCE_SINGLE, SOURCE_DUAL, SOURCE_UNKNOWN)

# Zout op een verpakking is NaCl; natrium is daar het aandeel natrium van.
SALT_TO_SODIUM = 0.393


@dataclass
class Product:
    """Eén product uit de database, met de waarden per eenheid.

    Een "eenheid" is wat je in één keer neemt: één gel, één sachet/schepje
    poeder, één reep. ``carbs_g``, ``sodium_mg`` en ``caffeine_mg`` gelden dus
    per eenheid, niet per 100 g.

    ``serving_ml`` is voor gels het volume van de gel en voor drank het
    *aanbevolen* mengvolume van de fabrikant. Dat mengvolume is informatief:
    de planner rekent de concentratie uit over het **totale** vochtvolume van
    het plan, niet per product — een geconcentreerd sachet dat je aanvult met
    extra water is prima.
    """

    name: str
    kind: str = KIND_GEL
    carbs_g: float = 0.0
    source: str = SOURCE_UNKNOWN
    ratio: str = ""
    sodium_mg: float = 0.0
    caffeine_mg: float = 0.0
    serving_ml: float | None = None
    serving_g: float | None = None
    note: str = ""
    active: bool = True

    @property
    def is_drink(self) -> bool:
        return self.kind == KIND_DRINK

    @property
    def effective_source(self) -> str:
        """De bron waar de planner mee rekent: ``onbekend`` telt als single."""
        return SOURCE_DUAL if self.source == SOURCE_DUAL else SOURCE_SINGLE

    @property
    def unit_label(self) -> str:
        """Hoe één eenheid heet in de meeneemlijst ('gel', 'sachet', 'reep')."""
        if self.kind == KIND_DRINK:
            return "sachet/schepje"
        if self.kind == KIND_SOLID:
            return "portie"
        return "gel"

    def as_text(self) -> str:
        """Compacte productregel voor de tijdlijn en het logboek."""
        delen = [f"{self.carbs_g:.0f} g kh"]
        if self.sodium_mg:
            delen.append(f"{self.sodium_mg:.0f} mg natrium")
        if self.caffeine_mg:
            delen.append(f"{self.caffeine_mg:.0f} mg cafeïne")
        return f"{self.name} ({', '.join(delen)})"


# De producten die ik gebruik. Waarden van de verpakking; controleer ze bij een
# nieuwe batch — recepturen veranderen.
DEFAULT_PRODUCTS: list[Product] = [
    Product(
        name="SIS Go Isotonic gel",
        kind=KIND_GEL, carbs_g=22.0,
        source=SOURCE_SINGLE, ratio="maltodextrine",
        sodium_mg=10.0, caffeine_mg=0.0, serving_ml=60.0,
        note="Isotoon: kan zonder water. Single-source, dus plafond 60 g/uur.",
    ),
    Product(
        name="SIS Go Isotonic gel + cafeïne",
        kind=KIND_GEL, carbs_g=22.0,
        source=SOURCE_SINGLE, ratio="maltodextrine",
        sodium_mg=10.0, caffeine_mg=75.0, serving_ml=60.0,
        note="Cafeïnevariant; plan hem in de tweede helft.",
    ),
    Product(
        name="SIS Beta Fuel gel",
        kind=KIND_GEL, carbs_g=40.0,
        source=SOURCE_DUAL, ratio="1:0,8 (glucose:fructose)",
        sodium_mg=35.0, caffeine_mg=0.0, serving_ml=60.0,
        note="Dual-source: tilt het plafond naar 90 g/uur.",
    ),
    Product(
        name="SIS Beta Fuel gel + cafeïne",
        kind=KIND_GEL, carbs_g=40.0,
        source=SOURCE_DUAL, ratio="1:0,8 (glucose:fructose)",
        sodium_mg=35.0, caffeine_mg=200.0, serving_ml=60.0,
        note="200 mg cafeïne per gel — let op het plafond van 3 mg/kg.",
    ),
    Product(
        name="SIS Beta Fuel drank (sachet)",
        kind=KIND_DRINK, carbs_g=80.0,
        source=SOURCE_DUAL, ratio="1:0,8 (glucose:fructose)",
        sodium_mg=78.0, caffeine_mg=0.0, serving_ml=500.0, serving_g=84.0,
        note="Controleer de actuele waarde op de verpakking. Geconcentreerd "
             "bedoeld: meng met extra water als het plan om meer vocht vraagt.",
    ),
    Product(
        name="Lidl HealthyFit Isotonic (schepje)",
        kind=KIND_DRINK, carbs_g=30.0,
        source=SOURCE_UNKNOWN,
        ratio="maltodextrine + dextrose + fructose + isomaltulose, verhouding "
              "niet vermeld",
        sodium_mg=79.0, caffeine_mg=0.0, serving_ml=500.0, serving_g=35.0,
        note="Bevat fructose, maar zonder ratio op de verpakking; telt daarom "
             "conservatief als single-source (plafond 60 g/uur). Per schepje "
             "ook 73 mg kalium en 80 mg calcium; natrium omgerekend uit "
             "0,20 g zout.",
    ),
]

PRODUCT_SCHEMA = """
CREATE TABLE IF NOT EXISTS nutrition_products (
    name        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    carbs_g     REAL,
    source      TEXT,
    ratio       TEXT,
    sodium_mg   REAL,
    caffeine_mg REAL,
    serving_ml  REAL,
    serving_g   REAL,
    note        TEXT,
    active      INTEGER,
    sort_order  INTEGER,
    updated_at  TEXT
);
"""

# Kolomvolgorde van de bewerkbare tabel in de UI (kolom -> label).
EDITOR_COLUMNS = {
    "name": "Product",
    "kind": "Type",
    "carbs_g": "Koolhydraten (g)",
    "source": "Bron",
    "ratio": "Ratio / samenstelling",
    "sodium_mg": "Natrium (mg)",
    "caffeine_mg": "Cafeïne (mg)",
    "serving_ml": "Volume (ml)",
    "serving_g": "Portie (g)",
    "note": "Opmerking",
    "active": "In voorraad",
}


def ensure_table(conn: sqlite3.Connection) -> None:
    """Maak de producttabel aan en vul hem bij eerste gebruik (idempotent)."""
    conn.executescript(PRODUCT_SCHEMA)
    leeg = conn.execute("SELECT COUNT(*) FROM nutrition_products").fetchone()[0] == 0
    if leeg:
        save_products(conn, DEFAULT_PRODUCTS)
    conn.commit()


def load_products(conn: sqlite3.Connection, only_active: bool = False) -> list[Product]:
    """Alle producten in de ingestelde volgorde, als :class:`Product`-objecten."""
    ensure_table(conn)
    vraag = "SELECT * FROM nutrition_products"
    if only_active:
        vraag += " WHERE active = 1"
    vraag += " ORDER BY sort_order, name"
    velden = set(Product.__dataclass_fields__)
    cur = conn.execute(vraag)
    kolommen = [d[0] for d in cur.description]
    out = []
    for row in cur.fetchall():
        kwargs = {k: v for k, v in zip(kolommen, row) if k in velden}
        kwargs["active"] = bool(kwargs.get("active", 1))
        out.append(Product(**kwargs))
    return out


def save_products(conn: sqlite3.Connection, products: list[Product]) -> int:
    """Schrijf de complete productlijst weg (vervangt de bestaande tabel).

    De lijst uit de editor is de waarheid: producten die de gebruiker heeft
    verwijderd verdwijnen dus ook echt. Producten zonder naam worden
    overgeslagen (lege regels uit de editor). Geeft het aantal opgeslagen
    producten terug.
    """
    conn.executescript(PRODUCT_SCHEMA)
    conn.execute("DELETE FROM nutrition_products")
    nu = datetime.now().isoformat(timespec="seconds")
    n = 0
    for i, p in enumerate(products):
        if not (p.name or "").strip():
            continue
        conn.execute(
            "INSERT OR REPLACE INTO nutrition_products (name, kind, carbs_g, "
            "source, ratio, sodium_mg, caffeine_mg, serving_ml, serving_g, "
            "note, active, sort_order, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.name.strip(), p.kind, float(p.carbs_g or 0), p.source,
             p.ratio or "", float(p.sodium_mg or 0), float(p.caffeine_mg or 0),
             p.serving_ml, p.serving_g, p.note or "", int(bool(p.active)), i, nu),
        )
        n += 1
    conn.commit()
    return n


def reset_products(conn: sqlite3.Connection) -> int:
    """Zet de productlijst terug op :data:`DEFAULT_PRODUCTS`."""
    return save_products(conn, DEFAULT_PRODUCTS)


def products_dataframe(products: list[Product]) -> pd.DataFrame:
    """De productlijst als DataFrame met de editor-kolommen, voor de UI."""
    if not products:
        return pd.DataFrame(columns=list(EDITOR_COLUMNS))
    df = pd.DataFrame([asdict(p) for p in products])
    return df[list(EDITOR_COLUMNS)]


def products_from_dataframe(df: pd.DataFrame) -> list[Product]:
    """Bouw :class:`Product`-objecten uit een (bewerkte) editor-tabel terug."""
    out = []
    for _, row in df.iterrows():
        naam = str(row.get("name") or "").strip()
        if not naam:
            continue
        kind = str(row.get("kind") or KIND_GEL)
        bron = str(row.get("source") or SOURCE_UNKNOWN)
        out.append(Product(
            name=naam,
            kind=kind if kind in KINDS else KIND_GEL,
            carbs_g=float(row.get("carbs_g") or 0),
            source=bron if bron in SOURCES else SOURCE_UNKNOWN,
            ratio=str(row.get("ratio") or ""),
            sodium_mg=float(row.get("sodium_mg") or 0),
            caffeine_mg=float(row.get("caffeine_mg") or 0),
            serving_ml=_optional_float(row.get("serving_ml")),
            serving_g=_optional_float(row.get("serving_g")),
            note=str(row.get("note") or ""),
            active=bool(row.get("active", True)),
        ))
    return out


def _optional_float(value) -> float | None:
    """Een leeg editor-veld wordt None, een getal een float."""
    if value is None or value == "" or pd.isna(value):
        return None
    return float(value)


def has_dual_source(products: list[Product]) -> bool:
    """Zit er minstens één dual-source product in deze selectie?"""
    return any(p.effective_source == SOURCE_DUAL for p in products)


def salt_g_to_sodium_mg(salt_g: float) -> float:
    """Reken zout (g NaCl, zoals op de verpakking) om naar natrium (mg)."""
    return salt_g * 1000 * SALT_TO_SODIUM
