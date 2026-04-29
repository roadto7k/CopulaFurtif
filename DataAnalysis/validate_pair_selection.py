"""
validate_pair_selection.py
==========================
Sélection de paires (ADF / KSS) avec export PDF + comparaison Tables 8-9 du papier.

Usage :
    # Données article (Tadi & Witzany 2025, 5-min, toutes les semaines)
    python validate_pair_selection.py --source article_5min

    # Sous-ensemble de semaines
    python validate_pair_selection.py --source article_5min --weeks 80-104

    # Binance live (cache local)
    python validate_pair_selection.py --source binance \
        --symbols BTCUSDT ETHUSDT LTCUSDT BNBUSDT ADAUSDT \
        --start 2024-01-01 --end 2025-01-01 --interval 1h

    # CSV locaux
    python validate_pair_selection.py --source csv

Options communes :
    --test       adf | kss | both        (défaut: both)
    Ranking:
    fixed paper-faithful ranking:
    Kendall tau between BTCUSDT log-returns and each accepted altcoin's log-returns.
                                          (défaut: kendall_returns — match Table 8 paper)
    --formation  semaines de formation    (défaut: 3)
    --trading    semaine de trading       (défaut: 1)
    --adf-alpha  seuil ADF                (défaut: 0.10)
    --kss-crit   valeur critique KSS      (défaut: -1.92)
    --output     chemin du PDF            (défaut: pair_selection_report.pdf)
    --verbose    affiche tous les coins et leurs stats

Note ranking :
    kendall_returns       → τ(r_BTC, r_alt) sur log-returns [DEFAULT, paper-faithful]
                            Reproduit Table 8 sur Week 1 (LTC=0.598, BCH=0.557).
                            Statistiquement correct : returns sont stationnaires.

    kendall_prices        → τ(P_BTC, P_alt) sur prix bruts
                            ⚠ Corrélation spurielle sur séries non-stationnaires.

    kendall_spread_pair   → τ(S_i, S_j) per Eq. 33
                            Lecture alternative qui contredit le texte du papier.

    kendall_spread_ref    → τ(S_alt, P_BTC)
                            Hybride sans justification théorique.
"""

import sys, os, argparse, textwrap
from datetime import datetime
import pandas as pd
import numpy as np

# ROOT = dossier contenant ce script.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
if not os.path.isdir(os.path.join(ROOT, 'dash_bot')):
    ROOT = os.path.abspath(os.path.join(ROOT, '..'))
    sys.path.insert(0, ROOT)
    if not os.path.isdir(os.path.join(ROOT, 'dash_bot')):
        raise RuntimeError(
            'dash_bot/ introuvable. Placez validate_pair_selection.py '
            'dans CopulaFurtif-main/ (à côté de dash_bot/).'
        )

from dash_bot.core.selection import select_pair_from_formation_window
from dash_bot.data.sources     import fetch_prices_cached, load_prices_csv
from dash_bot.data.cleaning    import clean_prices_basic

# ── reportlab ─────────────────────────────────────────────────────────────────
from reportlab.lib.pagesizes   import A4
from reportlab.lib.units       import mm
from reportlab.lib             import colors
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)

# ── constantes article ─────────────────────────────────────────────────────────
ARTICLE_COINS = [
    "BTCUSDT","ETHUSDT","BCHUSDT","XRPUSDT","EOSUSDT","LTCUSDT","TRXUSDT",
    "ETCUSDT","LINKUSDT","XLMUSDT","ADAUSDT","XMRUSDT","DASHUSDT","ZECUSDT",
    "XTZUSDT","ATOMUSDT","BNBUSDT","ONTUSDT","IOTAUSDT","BATUSDT",
]
ARTICLE_REF  = "BTCUSDT"
ARTICLE_CSV  = os.path.join(ROOT, "DataAnalysis", "data", "article_5min.csv")


# ════════════════════════════════════════════════════════════════════════════
#  PAPER GROUND TRUTH — Tables 8 et 9 (Tadi & Witzany 2025)
#  Format : { week: { (test, freq) : "C1-C2" } }
#  test ∈ {"adf", "kss"}, freq ∈ {"5m", "1h"}
# ════════════════════════════════════════════════════════════════════════════
PAPER_PAIRS = {
    # Table 8 — Weeks 1-52
    1:  {("adf","5m"):"LTC-BCH", ("kss","5m"):"LTC-BCH", ("adf","1h"):"ETH-LTC", ("kss","1h"):"BCH-ETC"},
    2:  {("adf","5m"):"LTC-BCH", ("kss","5m"):"LTC-XRP", ("adf","1h"):"LTC-BCH", ("kss","1h"):"LTC-BCH"},
    3:  {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-LTC", ("kss","1h"):"ETH-LTC"},
    4:  {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-LTC", ("kss","1h"):"ETH-ETC"},
    5:  {("adf","5m"):"LTC-EOS", ("kss","5m"):"LTC-BCH", ("adf","1h"):"LTC-EOS", ("kss","1h"):"LTC-BCH"},
    6:  {("adf","5m"):"TRX-XRP", ("kss","5m"):"LTC-BCH", ("adf","1h"):"LINK-TRX",("kss","1h"):"LTC-BCH"},
    7:  {("adf","5m"):"TRX-LINK",("kss","5m"):"LTC-BCH", ("adf","1h"):"LINK-TRX",("kss","1h"):"LTC-BCH"},
    8:  {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-BCH", ("adf","1h"):"ETH-LTC", ("kss","1h"):"ETH-BCH"},
    9:  {("adf","5m"):"ETH-LTC", ("kss","5m"):"LTC-BCH", ("adf","1h"):"ETH-LTC", ("kss","1h"):"LTC-BNB"},
    10: {("adf","5m"):"BCH-BNB", ("kss","5m"):"LTC-BCH", ("adf","1h"):"ETH-BCH", ("kss","1h"):"LTC-BCH"},
    11: {("adf","5m"):"LTC-BCH", ("kss","5m"):"LTC-BCH", ("adf","1h"):"LTC-BCH", ("kss","1h"):"LTC-BCH"},
    12: {("adf","5m"):"ADA-ATOM",("kss","5m"):"LTC-LINK",("adf","1h"):"ATOM-ADA",("kss","1h"):"BCH-ATOM"},
    13: {("adf","5m"):None,      ("kss","5m"):"ADA-XTZ", ("adf","1h"):"BAT-ONT", ("kss","1h"):"ADA-XTZ"},
    14: {("adf","5m"):"ADA-EOS", ("kss","5m"):"LTC-BCH", ("adf","1h"):"ADA-EOS", ("kss","1h"):"LTC-ADA"},
    15: {("adf","5m"):"LTC-EOS", ("kss","5m"):"LTC-EOS", ("adf","1h"):"EOS-LTC", ("kss","1h"):"EOS-LTC"},
    16: {("adf","5m"):"BNB-TRX", ("kss","5m"):"TRX-XRP", ("adf","1h"):"BAT-IOTA",("kss","1h"):"TRX-BNB"},
    17: {("adf","5m"):"TRX-BNB", ("kss","5m"):"TRX-LTC", ("adf","1h"):"BNB-TRX", ("kss","1h"):"BCH-BNB"},
    18: {("adf","5m"):"LTC-TRX", ("kss","5m"):"BNB-TRX", ("adf","1h"):"BCH-LTC", ("kss","1h"):"TRX-IOTA"},
    19: {("adf","5m"):"ETH-TRX", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-TRX", ("kss","1h"):"DASH-BCH"},
    20: {("adf","5m"):"ETH-LINK",("kss","5m"):"ETH-ONT", ("adf","1h"):"ETH-LTC", ("kss","1h"):"BCH-EOS"},
    21: {("adf","5m"):"TRX-LTC", ("kss","5m"):"TRX-LTC", ("adf","1h"):"TRX-LTC", ("kss","1h"):"TRX-LTC"},
    22: {("adf","5m"):None,      ("kss","5m"):"BNB-LTC", ("adf","1h"):None,      ("kss","1h"):"LTC-ETC"},
    23: {("adf","5m"):"LINK-XMR",("kss","5m"):"BNB-LINK",("adf","1h"):None,      ("kss","1h"):"BNB-LINK"},
    24: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    25: {("adf","5m"):"LTC-BCH", ("kss","5m"):"BNB-LTC", ("adf","1h"):"LTC-EOS", ("kss","1h"):"LTC-BNB"},
    26: {("adf","5m"):"LTC-BCH", ("kss","5m"):"LTC-BCH", ("adf","1h"):"ETH-LTC", ("kss","1h"):"LTC-XRP"},
    27: {("adf","5m"):"BNB-XRP", ("kss","5m"):"BCH-ONT", ("adf","1h"):"BNB-DASH",("kss","1h"):"XRP-DASH"},
    28: {("adf","5m"):"ETH-LINK",("kss","5m"):"ETH-LINK",("adf","1h"):"ETH-BCH", ("kss","1h"):"ETH-BCH"},
    29: {("adf","5m"):"BNB-LTC", ("kss","5m"):"BNB-LTC", ("adf","1h"):"BNB-LTC", ("kss","1h"):"BNB-LTC"},
    30: {("adf","5m"):"BNB-LTC", ("kss","5m"):"BNB-LTC", ("adf","1h"):"BNB-EOS", ("kss","1h"):"BNB-EOS"},
    31: {("adf","5m"):"ETH-LTC", ("kss","5m"):"LTC-LINK",("adf","1h"):"ETH-LINK",("kss","1h"):"LTC-LINK"},
    32: {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-LINK",("kss","1h"):"ETH-LTC"},
    33: {("adf","5m"):"LTC-EOS", ("kss","5m"):"BNB-LTC", ("adf","1h"):"LTC-XRP", ("kss","1h"):"LTC-BNB"},
    34: {("adf","5m"):"XRP-EOS", ("kss","5m"):"BNB-XRP", ("adf","1h"):"EOS-XRP", ("kss","1h"):"EOS-LTC"},
    35: {("adf","5m"):"ETH-XRP", ("kss","5m"):"BNB-XRP", ("adf","1h"):"ETH-EOS", ("kss","1h"):"EOS-BNB"},
    36: {("adf","5m"):"BNB-XRP", ("kss","5m"):"LTC-XMR", ("adf","1h"):"BNB-XRP", ("kss","1h"):"LTC-XMR"},
    37: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-ETC", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-ETC"},
    38: {("adf","5m"):"ETH-BNB", ("kss","5m"):"BNB-LTC", ("adf","1h"):"ETH-LINK",("kss","1h"):"BCH-ONT"},
    39: {("adf","5m"):"DASH-ONT",("kss","5m"):"XRP-DASH",("adf","1h"):"DASH-ONT",("kss","1h"):"LTC-DASH"},
    40: {("adf","5m"):"LTC-BNB", ("kss","5m"):"DASH-TRX",("adf","1h"):"LTC-BNB", ("kss","1h"):"DASH-BNB"},
    41: {("adf","5m"):"ETC-BCH", ("kss","5m"):"ETC-BCH", ("adf","1h"):"ETC-DASH",("kss","1h"):"ETC-DASH"},
    42: {("adf","5m"):"BCH-ETC", ("kss","5m"):"BCH-ETC", ("adf","1h"):"BCH-ETC", ("kss","1h"):"BCH-ETC"},
    43: {("adf","5m"):"ETC-EOS", ("kss","5m"):"ETC-BCH", ("adf","1h"):"ETH-ETC", ("kss","1h"):"ETC-EOS"},
    44: {("adf","5m"):"ETH-ETC", ("kss","5m"):"ETH-LINK",("adf","1h"):"ETH-ETC", ("kss","1h"):"ETH-ETC"},
    45: {("adf","5m"):"ETC-EOS", ("kss","5m"):"ETC-EOS", ("adf","1h"):"EOS-ETC", ("kss","1h"):"EOS-ETC"},
    46: {("adf","5m"):"ETC-EOS", ("kss","5m"):"ETC-LINK",("adf","1h"):"ETC-EOS", ("kss","1h"):"ETC-EOS"},
    47: {("adf","5m"):"ETC-EOS", ("kss","5m"):"ETC-EOS", ("adf","1h"):"ETC-EOS", ("kss","1h"):"ETC-EOS"},
    48: {("adf","5m"):"LTC-BNB", ("kss","5m"):"LTC-ETC", ("adf","1h"):"ETH-LTC", ("kss","1h"):"LTC-DASH"},
    49: {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-ETC", ("kss","1h"):"ETH-ETC"},
    50: {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-LTC", ("kss","1h"):"ETH-LTC"},
    51: {("adf","5m"):"ETH-ETC", ("kss","5m"):"ETH-ETC", ("adf","1h"):"ETC-LTC", ("kss","1h"):"ETH-ETC"},
    52: {("adf","5m"):"ETC-LTC", ("kss","5m"):"LTC-EOS", ("adf","1h"):"LTC-EOS", ("kss","1h"):"LTC-EOS"},
    # Table 9 — Weeks 53-104
    53: {("adf","5m"):"EOS-ETC", ("kss","5m"):"LTC-EOS", ("adf","1h"):"EOS-XRP", ("kss","1h"):"EOS-XRP"},
    54: {("adf","5m"):"BNB-BAT", ("kss","5m"):"BNB-ETC", ("adf","1h"):None,      ("kss","1h"):"LTC-BNB"},
    55: {("adf","5m"):"XLM-DASH",("kss","5m"):"EOS-XLM", ("adf","1h"):"DASH-XLM",("kss","1h"):"EOS-LTC"},
    56: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-LTC"},
    57: {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    58: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    59: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BAT", ("kss","1h"):"ETH-BAT"},
    60: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-EOS"},
    61: {("adf","5m"):"ETH-BNB", ("kss","5m"):"BNB-ADA", ("adf","1h"):"ETH-BNB", ("kss","1h"):"BNB-ADA"},
    62: {("adf","5m"):"BNB-ONT", ("kss","5m"):"ETH-BNB", ("adf","1h"):"BNB-ONT", ("kss","1h"):"ETH-BNB"},
    63: {("adf","5m"):"BNB-LTC", ("kss","5m"):"ETH-BNB", ("adf","1h"):"BNB-LTC", ("kss","1h"):"LTC-BNB"},
    64: {("adf","5m"):"LINK-LTC",("kss","5m"):"LINK-LTC",("adf","1h"):"LINK-ONT",("kss","1h"):"XRP-LINK"},
    65: {("adf","5m"):"ADA-LTC", ("kss","5m"):"ETH-ADA", ("adf","1h"):"XRP-XLM", ("kss","1h"):"XRP-ADA"},
    66: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    67: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    68: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    69: {("adf","5m"):None,      ("kss","5m"):"BNB-BCH", ("adf","1h"):None,      ("kss","1h"):"BCH-XLM"},
    70: {("adf","5m"):"BNB-ETC", ("kss","5m"):"BNB-LINK",("adf","1h"):"BNB-ETC", ("kss","1h"):"LINK-BNB"},
    71: {("adf","5m"):"BNB-LINK",("kss","5m"):"BNB-ETC", ("adf","1h"):"LINK-ETC",("kss","1h"):"ETC-BNB"},
    72: {("adf","5m"):"BNB-LINK",("kss","5m"):"BNB-LINK",("adf","1h"):"LINK-EOS",("kss","1h"):"BNB-LINK"},
    73: {("adf","5m"):"ETC-XRP", ("kss","5m"):"ETC-XRP", ("adf","1h"):"DASH-ETC",("kss","1h"):"EOS-DASH"},
    74: {("adf","5m"):"BNB-DASH",("kss","5m"):"ETH-BNB", ("adf","1h"):"ONT-ZEC", ("kss","1h"):"BNB-DASH"},
    75: {("adf","5m"):"ZEC-XTZ", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ZEC-BCH", ("kss","1h"):"ETH-BNB"},
    76: {("adf","5m"):"ETH-ZEC", ("kss","5m"):"ETH-ZEC", ("adf","1h"):"ETH-XTZ", ("kss","1h"):"ETH-ZEC"},
    77: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    78: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-BNB", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-BNB"},
    79: {("adf","5m"):"BNB-ADA", ("kss","5m"):"BNB-ADA", ("adf","1h"):"BNB-LTC", ("kss","1h"):"BNB-LTC"},
    80: {("adf","5m"):"BNB-ADA", ("kss","5m"):"BNB-ADA", ("adf","1h"):"BNB-LTC", ("kss","1h"):"BNB-LTC"},
    81: {("adf","5m"):"ETH-LTC", ("kss","5m"):"LTC-ADA", ("adf","1h"):"ETH-LTC", ("kss","1h"):"LTC-LINK"},
    82: {("adf","5m"):"ADA-LTC", ("kss","5m"):"ADA-LTC", ("adf","1h"):"LTC-XRP", ("kss","1h"):"LTC-XRP"},
    83: {("adf","5m"):"ADA-DASH",("kss","5m"):"ADA-DASH",("adf","1h"):"LTC-DASH",("kss","1h"):"LTC-DASH"},
    84: {("adf","5m"):"ETH-ADA", ("kss","5m"):"ETH-ADA", ("adf","1h"):"ETH-LTC", ("kss","1h"):"ETH-LTC"},
    85: {("adf","5m"):"ETH-LINK",("kss","5m"):"ETH-LINK",("adf","1h"):"ETH-IOTA",("kss","1h"):"IOTA-DASH"},
    86: {("adf","5m"):"BAT-BNB", ("kss","5m"):"DASH-BAT",("adf","1h"):"BAT-IOTA",("kss","1h"):"BAT-DASH"},
    87: {("adf","5m"):"BNB-DASH",("kss","5m"):"ETC-BAT", ("adf","1h"):"BNB-ETC", ("kss","1h"):"BAT-ETC"},
    88: {("adf","5m"):"BNB-BAT", ("kss","5m"):"DASH-ETC",("adf","1h"):"BNB-TRX", ("kss","1h"):"DASH-BAT"},
    89: {("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-DASH",("adf","1h"):"ETH-DASH",("kss","1h"):"ETH-DASH"},
    90: {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-DASH",("kss","1h"):"ETH-DASH"},
    91: {("adf","5m"):"ETH-BAT", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-BAT", ("kss","1h"):"ETH-DASH"},
    92: {("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-BAT", ("kss","1h"):"ETH-BAT"},
    93: {("adf","5m"):"LTC-BNB", ("kss","5m"):"LTC-BNB", ("adf","1h"):"BNB-LTC", ("kss","1h"):"BNB-LTC"},
    94: {("adf","5m"):"BCH-BAT", ("kss","5m"):"BCH-BAT", ("adf","1h"):"BCH-DASH",("kss","1h"):"BCH-DASH"},
    95: {("adf","5m"):"ETH-BCH", ("kss","5m"):"BCH-ONT", ("adf","1h"):"ETH-BCH", ("kss","1h"):"BCH-DASH"},
    96: {("adf","5m"):"ETH-ADA", ("kss","5m"):"ETH-ADA", ("adf","1h"):"ETH-ADA", ("kss","1h"):"ETH-ADA"},
    97: {("adf","5m"):"ETH-ADA", ("kss","5m"):"ETH-ADA", ("adf","1h"):"ETH-BAT", ("kss","1h"):"ETH-ADA"},
    98: {("adf","5m"):"ADA-ETC", ("kss","5m"):"ETH-ADA", ("adf","1h"):"ADA-BAT", ("kss","1h"):"ETH-ADA"},
    99: {("adf","5m"):"ADA-TRX", ("kss","5m"):"ETH-ETC", ("adf","1h"):"XTZ-BAT", ("kss","1h"):"ETH-ATOM"},
    100:{("adf","5m"):"ETH-LTC", ("kss","5m"):"ETH-BCH", ("adf","1h"):"ETH-BCH", ("kss","1h"):"ETH-ONT"},
    101:{("adf","5m"):"ETH-LINK",("kss","5m"):"LINK-ATOM",("adf","1h"):None,      ("kss","1h"):"LINK-BCH"},
    102:{("adf","5m"):"ETH-BNB", ("kss","5m"):"ETH-LTC", ("adf","1h"):"ETH-BNB", ("kss","1h"):"ETH-LTC"},
    103:{("adf","5m"):"BNB-LINK",("kss","5m"):"BNB-LINK",("adf","1h"):"BNB-LINK",("kss","1h"):"BNB-LINK"},
    104:{("adf","5m"):"LINK-BCH",("kss","5m"):"LINK-ADA",("adf","1h"):"LINK-DASH",("kss","1h"):"LINK-ONT"},
}


def _normalize_pair(pair_str):
    """Convertit 'LTC-BCH' en frozenset({'LTC','BCH'}) pour comparaison ordre-indep."""
    if pair_str is None or pair_str == "—" or not isinstance(pair_str, str):
        return None
    parts = pair_str.split("-")
    if len(parts) != 2:
        return None
    return frozenset(p.replace("USDT", "").strip() for p in parts)


def _compare_with_paper(week, test, freq, our_pair):
    """
    Retourne ("EXACT" | "PARTIAL" | "MISS" | "NA", paper_pair_str)
      EXACT   = même paire (ordre-indep)
      PARTIAL = un coin commun
      MISS    = aucun coin commun
      NA      = pas de paire référence dans Tables 8-9 (papier n'a pas tradé)
    """
    if week not in PAPER_PAIRS:
        return ("NA", "—")

    paper_str = PAPER_PAIRS[week].get((test, freq))
    if paper_str is None:
        return ("NA", "—")

    paper_set = _normalize_pair(paper_str)
    our_set = _normalize_pair(our_pair)

    if our_set is None or paper_set is None:
        return ("MISS", paper_str)

    if our_set == paper_set:
        return ("EXACT", paper_str)
    elif len(our_set & paper_set) >= 1:
        return ("PARTIAL", paper_str)
    else:
        return ("MISS", paper_str)


def _load_article_csv():
    """Charge article_5min.csv directement."""
    if not os.path.exists(ARTICLE_CSV):
        raise FileNotFoundError(
            "Fichier introuvable : " + ARTICLE_CSV + chr(10) +
            "Placez le fichier MOESM1 dans DataAnalysis/data/article_5min.csv"
        )
    df = pd.read_csv(ARTICLE_CSV, index_col=0, parse_dates=True)
    return df.sort_index().replace(0, np.nan)


# ── chargement données ─────────────────────────────────────────────────────────
def load_data(args):
    """Charge les prix selon la source choisie, identique au dashboard."""
    src = args.source

    if src == "article_5min":
        prices   = _load_article_csv()
        errors   = {}
        ref      = ARTICLE_REF
        symbols  = [c for c in ARTICLE_COINS if c in prices.columns]
        interval = "5m"

    elif src == "binance":
        symbols  = args.symbols or ARTICLE_COINS
        ref      = args.ref or ARTICLE_REF
        interval = args.interval or "1h"
        prices, errors = fetch_prices_cached(
            symbols, args.start, args.end, interval, source="binance"
        )

    else:  # csv
        from DataAnalysis.config import DATA_PATH
        prices = load_prices_csv(DATA_PATH)
        prices = prices.loc[
            (prices.index >= pd.to_datetime(args.start)) &
            (prices.index <= pd.to_datetime(args.end))
        ]
        interval = args.interval or "1h"
        if interval != "1h":
            rule = {"5m":"5T","15m":"15T","4h":"4H","1d":"1D"}.get(interval)
            if rule:
                prices = prices.resample(rule).last()
        symbols  = args.symbols or [c for c in prices.columns]
        ref      = args.ref or ARTICLE_REF
        errors   = {}

    prices = prices.replace(0, np.nan)
    prices = clean_prices_basic(prices)

    if errors:
        for sym, msg in errors.items():
            print(f"  ⚠ {sym}: {msg}")

    print(f"  Données : {prices.shape[0]:,} barres  "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    return prices, ref, [s for s in symbols if s in prices.columns], interval


# ── cycles ─────────────────────────────────────────────────────────────────────
def build_cycles(prices, formation_w, trading_w, n_cycles):
    origin = prices.index[0].normalize()
    cycles = []
    for i in range(n_cycles):
        fs = origin + pd.Timedelta(weeks=i)
        fe = fs     + pd.Timedelta(weeks=formation_w)
        ts, te = fe, fe + pd.Timedelta(weeks=trading_w)
        fd = prices.loc[fs : fe - pd.Timedelta(minutes=1)]
        td = prices.loc[ts : te - pd.Timedelta(minutes=1)]
        if fd.empty or td.empty:
            continue
        cycles.append((i + 1, fd, td))
    return cycles


# ── boucle principale ──────────────────────────────────────────────────────────
def run(args):
    print("\n── Chargement des données ─────────────────────────────")
    prices, ref, candidates, interval = load_data(args)

    tests    = ["adf", "kss"] if args.test == "both" else [args.test]
    n_cycles = args.n_cycles or (104 if args.source == "article_5min" else 52)

    # interval normalisé pour matching paper
    freq_key = "5m" if interval in ("5m", "5T") else ("1h" if interval in ("1h", "1H") else None)

    print(f"\n── Paramètres ──────────────────────────────────────────")
    print(f"  Source      : {args.source}  |  interval : {interval}")
    print(f"  Ref asset   : {ref}")
    print(f"  Candidates  : {len(candidates)} coins")
    print(f"  Cycles      : {n_cycles}  (formation={args.formation}w, trading={args.trading}w)")
    print(f"  Tests       : {', '.join(t.upper() for t in tests)}")
    print("  Rank method : Kendall tau on log-returns vs BTCUSDT")
    print(f"  ADF alpha   : {args.adf_alpha}  |  KSS crit : {args.kss_crit}")
    print(f"  Paper match : {'enabled (' + freq_key + ')' if freq_key else 'disabled (interval not in {5m, 1h})'}")

    cycles = build_cycles(prices, args.formation, args.trading, n_cycles)
    print(f"\n── Sélection paires ({len(cycles)} cycles) ─────────────────")

    week_range = None
    if args.weeks:
        a, b = args.weeks.split("-")
        week_range = set(range(int(a), int(b) + 1))

    records = []
    for (cid, fd, td) in cycles:
        if week_range and cid not in week_range:
            continue
        if args.verbose:
            print(f"\n  Cycle {cid}  {fd.index[0].date()} → {fd.index[-1].date()}")
        row = {
            "week":        cid,
            "form_start":  str(fd.index[0].date()),
            "form_end":    str(fd.index[-1].date()),
            "trade_start": str(td.index[0].date()),
        }
        for t in tests:
            res, summary, spreads, betas, ranked = select_pair_from_formation_window(
                form_data=fd,
                ref=ref,
                candidates=candidates,
                cointegration_test=t,
                adf_alpha=args.adf_alpha,
                kss_crit=args.kss_crit,
                min_obs=50,
                verbose=args.verbose,
            )

            our_pair = res["pair"] if res else "—"

            row[f"{t}_pair"] = our_pair
            row[f"{t}_s1"] = round(res["stat1"], 3) if res and np.isfinite(res["stat1"]) else np.nan
            row[f"{t}_s2"] = round(res["stat2"], 3) if res and np.isfinite(res["stat2"]) else np.nan
            row[f"{t}_tau1"] = round(res["tau1"], 4) if res and np.isfinite(res["tau1"]) else np.nan
            row[f"{t}_tau2"] = round(res["tau2"], 4) if res and np.isfinite(res["tau2"]) else np.nan

            # Paper match (only meaningful for article_5min/1h)
            if freq_key:
                match_status, paper_pair = _compare_with_paper(cid, t, freq_key, our_pair)
                row[f"{t}_paper"] = paper_pair
                row[f"{t}_match"] = match_status
            else:
                row[f"{t}_paper"] = "—"
                row[f"{t}_match"] = "NA"

        records.append(row)
        if not args.verbose and cid % 10 == 0:
            print(f"  ... cycle {cid}/{n_cycles}")

    df = pd.DataFrame(records)
    print(f"\n  {len(df)} cycles traités.\n")

    # ── Résumé console match-rate ─────────────────────────────────────────────
    if freq_key:
        print("── Match rate vs Paper Tables 8-9 ──────────────────────")
        for t in tests:
            sub = df[df[f"{t}_match"] != "NA"]
            n = len(sub)
            if n == 0:
                continue
            exact = (sub[f"{t}_match"] == "EXACT").sum()
            partial = (sub[f"{t}_match"] == "PARTIAL").sum()
            miss = (sub[f"{t}_match"] == "MISS").sum()
            print(f"  {t.upper():4} : EXACT {exact}/{n} ({100*exact/n:.1f}%)  "
                  f"PARTIAL {partial}/{n} ({100*partial/n:.1f}%)  "
                  f"MISS {miss}/{n} ({100*miss/n:.1f}%)")

    # ── PDF ───────────────────────────────────────────────────────────────────
    out = args.output or "pair_selection_report.pdf"
    build_pdf(df, args, tests, ref, candidates, interval, n_cycles, freq_key, out)
    print(f"\n✓ PDF généré → {out}")
    return df


# ── construction PDF ──────────────────────────────────────────────────────────
def build_pdf(df, args, tests, ref, candidates, interval, n_cycles, freq_key, out_path):

    # ── couleurs & styles Bloomberg terminal ──────────────────────────────────
    BG       = colors.HexColor("#0d1117")
    CARD     = colors.HexColor("#161b22")
    BORDER   = colors.HexColor("#30363d")
    ACCENT   = colors.HexColor("#f78166")
    GREEN    = colors.HexColor("#3fb950")
    YELLOW   = colors.HexColor("#d29922")
    RED      = colors.HexColor("#f85149")
    BLUE     = colors.HexColor("#58a6ff")
    FG       = colors.HexColor("#e6edf3")
    FG_DIM   = colors.HexColor("#8b949e")
    HEADER_BG= colors.HexColor("#21262d")

    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm,  bottomMargin=15*mm,
    )

    styles = getSampleStyleSheet()

    def sty(name, **kw):
        base = dict(fontName="Helvetica", fontSize=9,
                    textColor=FG, leading=13, backColor=None)
        base.update(kw)
        return ParagraphStyle(name, **base)

    s_title  = sty("title",  fontSize=18, textColor=ACCENT,
                   fontName="Helvetica-Bold", spaceAfter=4)
    s_sub    = sty("sub",    fontSize=10, textColor=FG_DIM, spaceAfter=10)
    s_h1     = sty("h1",     fontSize=13, textColor=ACCENT,
                   fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=6)
    s_h2     = sty("h2",     fontSize=10, textColor=YELLOW,
                   fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    s_body   = sty("body",   fontSize=8,  textColor=FG,     leading=12)
    s_note   = sty("note",   fontSize=7,  textColor=FG_DIM, leading=11)

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=BORDER, spaceAfter=6, spaceBefore=2)

    def kv(key, val, key_color=FG_DIM, val_color=FG):
        return Paragraph(
            f'<font color="{key_color}">{key}</font>  '
            f'<font color="{val_color}"><b>{val}</b></font>',
            s_body
        )

    def match_color(status):
        return {"EXACT": GREEN, "PARTIAL": YELLOW, "MISS": RED, "NA": FG_DIM}.get(status, FG_DIM)

    def match_glyph(status):
        return {"EXACT": "✓✓", "PARTIAL": "~", "MISS": "✗", "NA": "—"}.get(status, "—")

    story = []

    # ──────────────────────────────────────────────────────────────────────────
    # Page 1 : titre + config + résumé
    # ──────────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("Pair Selection Validation Report", s_title))
    story.append(Paragraph("Copula-based pairs trading — Tadi &amp; Witzany (2025)", s_sub))
    story.append(hr())

    story.append(Paragraph("Run Configuration", s_h1))
    story.append(Spacer(1, 2))

    src_label = {
        "article_5min": "Article MOESM1 — 5-min (2021-01-01 → 2023-01-19)",
        "binance":       "Binance SPOT (cache local)",
        "csv":           "CSV locaux",
    }.get(args.source, args.source)

    cfg = [
        ("Data source",    src_label),
        ("Interval",       interval),
        ("Reference asset",ref),
        ("Candidates",     f"{len(candidates)} coins  [{', '.join(c.replace('USDT','') for c in candidates[:8])}{'…' if len(candidates)>8 else ''}]"),
        ("Cycles",         f"{n_cycles}  (formation={args.formation}w, trading={args.trading}w, step=1w)"),
        ("Tests",          " + ".join(t.upper() for t in tests)),
        ("Ranking method", "Kendall tau on log-returns vs BTCUSDT"),
        ("ADF alpha",      str(args.adf_alpha)),
        ("KSS critical",   str(args.kss_crit)),
        ("Weeks filter",   args.weeks or "all"),
        ("Paper match",    f"enabled ({freq_key})" if freq_key else "disabled"),
        ("Generated",      datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]

    cfg_data = [[Paragraph(f'<font color="{FG_DIM}">{k}</font>', s_body),
                 Paragraph(f'<b>{v}</b>', s_body)] for k, v in cfg]

    cfg_table = Table(cfg_data, colWidths=[45*mm, None],
                      hAlign="LEFT", repeatRows=0)
    cfg_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), CARD),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [CARD, HEADER_BG]),
        ("GRID",          (0,0), (-1,-1), 0.3, BORDER),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(cfg_table)

    # résumé des résultats + match-rate
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("Results Summary", s_h1))
    story.append(hr())

    total = len(df)
    for t in tests:
        found = (df[f"{t}_pair"] != "—").sum()
        lbl = "P-Values" if t == "adf" else "t-stats (crit=-1.92)"
        story.append(kv(
            f"{t.upper()} ({lbl})",
            f"{found}/{total} pairs found  "
            f"({'%.1f'%(100*found/total if total else 0)}%)"
        ))
        story.append(Spacer(1, 2))

    # Match-rate global vs paper
    if freq_key:
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(f"Paper Reproduction Match-Rate ({freq_key})", s_h2))

        for t in tests:
            sub = df[df[f"{t}_match"] != "NA"]
            n = len(sub)
            if n == 0:
                continue
            exact = (sub[f"{t}_match"] == "EXACT").sum()
            partial = (sub[f"{t}_match"] == "PARTIAL").sum()
            miss = (sub[f"{t}_match"] == "MISS").sum()

            line = (
                f'<font color="{GREEN}"><b>EXACT</b> {exact}/{n} ({100*exact/n:.1f}%)</font>  '
                f'&nbsp;&nbsp;<font color="{YELLOW}"><b>PARTIAL</b> {partial}/{n} ({100*partial/n:.1f}%)</font>  '
                f'&nbsp;&nbsp;<font color="{RED}"><b>MISS</b> {miss}/{n} ({100*miss/n:.1f}%)</font>'
            )
            story.append(kv(f"{t.upper()}", "", val_color=FG))
            story.append(Paragraph(line, s_body))
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "<b>Reproduction methodology</b> (after correction of 3 bugs identified) : "
        "(1) β estimated via OLS through origin per Eq. 6, "
        "(2) ADF tested with regression='n' on demeaned spread per Eq. 7, "
        "(3) Kendall τ ranking computed on log-RETURNS not prices "
        "(only interpretation that reproduces Table 8 Week 1 LTC-BCH).",
        s_note
    ))

    story.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────────────
    # Pages suivantes : tables de résultats par test (avec colonne Paper Match)
    # ──────────────────────────────────────────────────────────────────────────
    for t in tests:
        lbl      = "P-Value" if t == "adf" else "t-stat"
        crit_str = f"α={args.adf_alpha}" if t == "adf" else f"crit={args.kss_crit}"
        stat_color = GREEN if t == "adf" else YELLOW

        story.append(Paragraph(
            f"{t.upper()} Unit-Root Test — {lbl}s  ({crit_str})", s_h1))
        story.append(Paragraph(
            f"Source: {src_label}  |  Ranking: Kendall log-returns vs BTCUSDT  |  Paper match: {'on' if freq_key else 'off'}",
            s_note))
        story.append(Spacer(1, 3))

        # en-tête (étendue avec colonnes paper si applicable)
        if freq_key:
            hdr = [
                Paragraph("<b>Wk</b>",          s_body),
                Paragraph("<b>Form start</b>",   s_body),
                Paragraph("<b>Trade start</b>",  s_body),
                Paragraph("<b>Our pair</b>",     s_body),
                Paragraph(f"<b>S1</b>",          s_body),
                Paragraph(f"<b>S2</b>",          s_body),
                Paragraph("<b>Paper pair</b>",   s_body),
                Paragraph("<b>Match</b>",        s_body),
            ]
            col_w = [9*mm, 22*mm, 22*mm, 26*mm, 16*mm, 16*mm, 26*mm, 14*mm]
        else:
            hdr = [
                Paragraph("<b>Wk</b>",          s_body),
                Paragraph("<b>Form start</b>",   s_body),
                Paragraph("<b>Form end</b>",     s_body),
                Paragraph("<b>Trade start</b>",  s_body),
                Paragraph("<b>Pair</b>",         s_body),
                Paragraph(f"<b>{lbl} S1</b>",    s_body),
                Paragraph(f"<b>{lbl} S2</b>",    s_body),
            ]
            col_w = [10*mm, 24*mm, 24*mm, 24*mm, 28*mm, 22*mm, 22*mm]

        rows = [hdr]
        for _, r in df.iterrows():
            pair = r[f"{t}_pair"]
            s1   = f"{r[f'{t}_s1']:.3f}" if pd.notna(r[f'{t}_s1']) else "—"
            s2   = f"{r[f'{t}_s2']:.3f}" if pd.notna(r[f'{t}_s2']) else "—"
            pair_para = Paragraph(
                f'<font color="{str(stat_color)}">{pair}</font>'
                if pair != "—" else f'<font color="{FG_DIM}">—</font>',
                s_body
            )

            if freq_key:
                paper_pair  = r[f"{t}_paper"]
                match_stat  = r[f"{t}_match"]
                mc          = match_color(match_stat)
                glyph       = match_glyph(match_stat)

                paper_para = Paragraph(
                    f'<font color="{BLUE}">{paper_pair}</font>'
                    if paper_pair != "—" else f'<font color="{FG_DIM}">—</font>',
                    s_body
                )
                match_para = Paragraph(
                    f'<font color="{mc}"><b>{glyph}</b></font>',
                    s_body
                )

                rows.append([
                    Paragraph(str(int(r["week"])),    s_body),
                    Paragraph(str(r["form_start"]),   s_body),
                    Paragraph(str(r["trade_start"]),  s_body),
                    pair_para,
                    Paragraph(s1, s_body),
                    Paragraph(s2, s_body),
                    paper_para,
                    match_para,
                ])
            else:
                rows.append([
                    Paragraph(str(int(r["week"])),    s_body),
                    Paragraph(str(r["form_start"]),   s_body),
                    Paragraph(str(r["form_end"]),     s_body),
                    Paragraph(str(r["trade_start"]),  s_body),
                    pair_para,
                    Paragraph(s1, s_body),
                    Paragraph(s2, s_body),
                ])

        tbl = Table(rows, colWidths=col_w, repeatRows=1, hAlign="LEFT")

        ts_cmds = [
            ("BACKGROUND",   (0,0), (-1,0), HEADER_BG),
            ("TEXTCOLOR",    (0,0), (-1,0), FG),
            ("GRID",         (0,0), (-1,-1), 0.3, BORDER),
            ("LEFTPADDING",  (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING",   (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ]
        for i, _ in enumerate(rows[1:], start=1):
            bg = CARD if i % 2 == 0 else BG
            ts_cmds.append(("BACKGROUND", (0,i), (-1,i), bg))

        tbl.setStyle(TableStyle(ts_cmds))
        story.append(tbl)
        story.append(PageBreak())

    # ──────────────────────────────────────────────────────────────────────────
    # Dernière page : side-by-side ADF vs KSS si les deux tournent
    # ──────────────────────────────────────────────────────────────────────────
    if len(tests) == 2:
        story.append(Paragraph("ADF vs KSS — Side-by-side", s_h1))
        story.append(Paragraph(
            "Comparaison directe des paires sélectionnées par les deux tests "
            "sur chaque cycle.", s_note))
        story.append(Spacer(1, 3))

        hdr2 = [
            Paragraph("<b>Wk</b>",           s_body),
            Paragraph("<b>Trade start</b>",   s_body),
            Paragraph("<b>ADF pair</b>",      s_body),
            Paragraph("<b>ADF match</b>",     s_body),
            Paragraph("<b>KSS pair</b>",      s_body),
            Paragraph("<b>KSS match</b>",     s_body),
            Paragraph("<b>ADF=KSS</b>",       s_body),
        ]
        rows2 = [hdr2]
        for _, r in df.iterrows():
            adf_p = r["adf_pair"]
            kss_p = r["kss_pair"]
            same  = adf_p == kss_p and adf_p != "—"
            same_str = "✓" if same else ""
            same_c   = GREEN if same else FG_DIM

            adf_match = r.get("adf_match", "NA")
            kss_match = r.get("kss_match", "NA")

            rows2.append([
                Paragraph(str(int(r["week"])), s_body),
                Paragraph(str(r["trade_start"]), s_body),
                Paragraph(f'<font color="{GREEN}">{adf_p}</font>' if adf_p != "—"
                          else f'<font color="{FG_DIM}">—</font>', s_body),
                Paragraph(f'<font color="{match_color(adf_match)}"><b>{match_glyph(adf_match)}</b></font>', s_body),
                Paragraph(f'<font color="{YELLOW}">{kss_p}</font>' if kss_p != "—"
                          else f'<font color="{FG_DIM}">—</font>', s_body),
                Paragraph(f'<font color="{match_color(kss_match)}"><b>{match_glyph(kss_match)}</b></font>', s_body),
                Paragraph(f'<font color="{same_c}"><b>{same_str}</b></font>', s_body),
            ])

        col_w2 = [9*mm, 22*mm, 28*mm, 16*mm, 28*mm, 16*mm, 18*mm]
        tbl2 = Table(rows2, colWidths=col_w2, repeatRows=1, hAlign="LEFT")
        ts2 = [
            ("BACKGROUND",   (0,0), (-1,0), HEADER_BG),
            ("GRID",         (0,0), (-1,-1), 0.3, BORDER),
            ("LEFTPADDING",  (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING",   (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("FONTSIZE",     (0,0), (-1,-1), 7.5),
        ]
        for i, _ in enumerate(rows2[1:], start=1):
            bg = CARD if i % 2 == 0 else BG
            ts2.append(("BACKGROUND", (0,i), (-1,i), bg))

        tbl2.setStyle(TableStyle(ts2))
        story.append(tbl2)

        # Légende
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(
            f'Match legend : '
            f'<font color="{GREEN}"><b>✓✓ EXACT</b></font> = same pair as paper&nbsp;&nbsp;'
            f'<font color="{YELLOW}"><b>~ PARTIAL</b></font> = one coin in common&nbsp;&nbsp;'
            f'<font color="{RED}"><b>✗ MISS</b></font> = no overlap&nbsp;&nbsp;'
            f'<font color="{FG_DIM}"><b>— NA</b></font> = paper had no pair this week',
            s_note
        ))

    # ── fond noir sur toutes les pages ────────────────────────────────────────
    def add_background(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.setFillColor(FG_DIM)
        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(15*mm, 8*mm,
            f"Pair Selection Report  |  {datetime.now().strftime('%Y-%m-%d')}  "
            f"|  Source: {args.source}  |  Page {doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_background, onLaterPages=add_background)


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pair selection validation + PDF report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    # source de données (identique au dashboard)
    p.add_argument("--source",   choices=["article_5min","binance","csv"],
                   default="article_5min")
    p.add_argument("--symbols",  nargs="+", default=None,
                   help="Liste de symboles (ex: BTCUSDT ETHUSDT LTCUSDT)")
    p.add_argument("--ref",      default=None, help="Asset de référence (défaut: BTCUSDT)")
    p.add_argument("--start",    default="2024-01-01")
    p.add_argument("--end",      default="2025-01-01")
    p.add_argument("--interval", default="1h",
                   choices=["1m","5m","15m","1h","4h","1d"])

    # paramètres cycles
    p.add_argument("--formation", type=int, default=3)
    p.add_argument("--trading",   type=int, default=1)
    p.add_argument("--n-cycles",  type=int, default=None,
                   help="Nombre max de cycles (défaut: 104 pour article_5min, 52 sinon)")
    p.add_argument("--weeks",     type=str, default=None,
                   help="Ex: '80-104' pour filtrer les semaines affichées")

    # tests & ranking
    p.add_argument("--test",      choices=["adf","kss","both"], default="both")
    p.add_argument("--adf-alpha", type=float, default=0.10)
    p.add_argument("--kss-crit",  type=float, default=-1.92)

    # sortie
    p.add_argument("--output",  default="pair_selection_report.pdf")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    run(args)