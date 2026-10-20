"""Date comune pentru laboratoarele 3–6.

Aceeași încărcare, aceeași curățare în straturi și același tabel de clienți ca în Laboratoarele 1–2
(laborator_1_2_regresie_clasificare.ipynb, secțiunile 2 și 3). Le punem aici ca fiecare laborator
să pornească din exact aceleași date, fără să repetăm codul.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
ZILE_FEREASTRA = 90                       # fereastra „viitor” (ca la Laboratoarele 1–2)
ARTEFACTE = Path("artefacte")             # aici se salvează tot ce citește aplicația Streamlit
(ARTEFACTE / "figuri").mkdir(parents=True, exist_ok=True)

# coduri de servicii/taxe/ajustări/vouchere (alese după descriere și preț, nu după regula „5 cifre”)
COD_SERVICIU = r"POST|DOT|M|C2|D|S|BANK CHARGES|ADJUST2?|AMAZONFEE|CRUK|B|TEST\d+|gift_0001_\d+"


def salveaza_fig(nume):
    """Salvează figura curentă în artefacte/figuri și o afișează."""
    plt.gcf().savefig(ARTEFACTE / "figuri" / f"{nume}.png", dpi=130, bbox_inches="tight")
    plt.show()


def incarca_brut(cale=None):
    """Citește datasetul. Identificatorii sunt text (sunt etichete, nu cantități); `Customer ID` vine ca `13085.0`, deci tăiem `.0`."""
    cale = cale or os.environ.get("LAB_CSV")
    if cale is None:
        import kagglehub
        folder = kagglehub.dataset_download("mashlyn/online-retail-ii-uci")
        cale = next((str(p) for p in Path(folder).rglob("*.csv")), None)
    assert cale is not None, "Nu am găsit niciun .csv"
    df = pd.read_csv(cale, dtype={"Invoice": str, "StockCode": str, "Customer ID": str}, parse_dates=["InvoiceDate"])
    df["Customer ID"] = df["Customer ID"].str.replace(r"\.0$", "", regex=True)
    return df


def curata(df):
    """Curățare în straturi: rândurile sunt mutate, nu șterse. Întoarce (tranzactii, retururi, administrative, ajustari_stoc, contabilitate)."""
    m1 = df["StockCode"].str.fullmatch(COD_SERVICIU, case=False) | df["Invoice"].str.startswith("A")
    administrative, df1 = df[m1], df[~m1]

    m2 = df1["Invoice"].str.startswith("C")
    retururi, df2 = df1[m2].copy(), df1[~m2]

    m3 = (df2["Price"] <= 0) | (df2["Quantity"] <= 0)
    ajustari_stoc, tranzactii = df2[m3], df2[~m3].copy()
    assert len(administrative) + len(retururi) + len(ajustari_stoc) + len(tranzactii) == len(df)

    tranzactii["Valoare"] = tranzactii["Quantity"] * tranzactii["Price"]
    retururi["Valoare"] = retururi["Quantity"] * retururi["Price"]
    contabilitate = pd.DataFrame({
        "strat": ["1. servicii/taxe/vouchere", "2. anulări (C)", "3. preț/cantitate ≤ 0", "rămase: vânzări curate"],
        "rânduri": [len(administrative), len(retururi), len(ajustari_stoc), len(tranzactii)],
    })
    contabilitate["% din total"] = (contabilitate["rânduri"] / len(df) * 100).round(2)
    return tranzactii, retururi, administrative, ajustari_stoc, contabilitate


def construieste_clienti(tranzactii, retururi):
    """Tabelul de clienți (un rând = un client), cu data de tăiere: variabilele din istoric, ținta din ultimele 90 de zile.
    Întoarce (clienti, CUTOFF, istoric_v, viitor_v): istoricul și viitorul vânzărilor clienților identificați."""
    final = tranzactii["InvoiceDate"].max()
    cutoff = final.normalize() - pd.Timedelta(days=ZILE_FEREASTRA)

    vanzari = tranzactii[tranzactii["Customer ID"].notna()]
    retur_cl = retururi[retururi["Customer ID"].notna()]
    istoric_v, viitor_v = vanzari[vanzari["InvoiceDate"] < cutoff], vanzari[vanzari["InvoiceDate"] >= cutoff]
    istoric_r, viitor_r = retur_cl[retur_cl["InvoiceDate"] < cutoff], retur_cl[retur_cl["InvoiceDate"] >= cutoff]

    g = istoric_v.groupby("Customer ID")
    clienti = pd.DataFrame({
        "recenta_zile":      (cutoff - g["InvoiceDate"].max()).dt.days,
        "vechime_zile":      (cutoff - g["InvoiceDate"].min()).dt.days,
        "frecventa":         g["Invoice"].nunique(),
        "produse_distincte": g["StockCode"].nunique(),
        "cantitate_totala":  g["Quantity"].sum(),
        "vanzari_brute":     g["Valoare"].sum(),
        "uk":                (g["Country"].first() == "United Kingdom").astype(int),
    })
    clienti["retururi_val"] = istoric_r.groupby("Customer ID")["Valoare"].sum().reindex(clienti.index).fillna(0)
    ultimele = istoric_v[istoric_v["InvoiceDate"] >= cutoff - pd.Timedelta(days=ZILE_FEREASTRA)]
    clienti["vanzari_ultimele_90z"] = ultimele.groupby("Customer ID")["Valoare"].sum().reindex(clienti.index).fillna(0)
    clienti["valoare_medie_factura"] = clienti["vanzari_brute"] / clienti["frecventa"]
    clienti["rata_retur"] = (clienti["retururi_val"].abs() / clienti["vanzari_brute"]).clip(upper=1)

    tinta = (viitor_v.groupby("Customer ID")["Valoare"].sum()
             .add(viitor_r.groupby("Customer ID")["Valoare"].sum(), fill_value=0))
    clienti["tinta_cheltuiala"] = tinta.reindex(clienti.index).fillna(0)
    clienti["a_cumparat"] = clienti.index.isin(viitor_v["Customer ID"])
    return clienti, cutoff, istoric_v, viitor_v


def transforma(c):
    """Variabile de intrare: scară originală sau log1p pentru cele cu coadă lungă (identic cu Laboratoarele 1–2)."""
    X = pd.DataFrame(index=c.index)
    for col in ["recenta_zile", "vechime_zile", "rata_retur", "uk"]:
        X[col] = c[col]
    for col in ["frecventa", "produse_distincte", "cantitate_totala", "vanzari_brute",
                "vanzari_ultimele_90z", "valoare_medie_factura"]:
        X["log_" + col] = np.log1p(c[col])
    X["log_retururi"] = np.log1p(c["retururi_val"].abs())
    return X


def descrieri_produse(tranzactii):
    """StockCode -> descrierea cea mai frecventă (un cod poate apărea cu descrieri ușor diferite)."""
    d = tranzactii.dropna(subset=["Description"])
    return d.groupby("StockCode")["Description"].agg(lambda s: s.value_counts().index[0])
