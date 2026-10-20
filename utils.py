"""Funcții comune pentru paginile Streamlit.

Aplicația doar CITEȘTE ce a salvat fiecare notebook în folderul `artefacte/`; nu recalculează nimic.
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artefacte"
alt.data_transformers.disable_max_rows()


# ───────────────────────── citire artefacte ─────────────────────────
def cere_artefacte(fisier="info.json", notebook="laborator_1_2_regresie_clasificare.ipynb"):
    """Oprește pagina cu un mesaj clar dacă notebook-ul laboratorului nu a fost rulat încă."""
    if not (ART / fisier).exists():
        st.error(f"Nu găsesc `artefacte/{fisier}`. Rulează întâi notebook-ul `{notebook}` "
                 "(complet, până la secțiunea Implementare) în acest folder, apoi reîncarcă pagina.")
        st.stop()


@st.cache_data
def info(fisier="info.json"):
    return json.loads((ART / fisier).read_text(encoding="utf-8"))


def info_optional(fisier):
    """Ca `info`, dar întoarce None dacă notebook-ul nu a fost rulat (pentru pagini care combină mai multe laboratoare)."""
    return info(fisier) if (ART / fisier).exists() else None


@st.cache_data
def tabel(nume, index_col=0):
    cale = ART / nume
    if not cale.exists():
        return None
    return pd.read_csv(cale, index_col=index_col)


@st.cache_data
def tabel_text(nume):
    """Citește un CSV cu toate coloanele ca text (pentru coduri de produs precum 85123A sau 00123)."""
    cale = ART / nume
    return pd.read_csv(cale, dtype=str) if cale.exists() else None


@st.cache_resource
def model(nume):
    return joblib.load(ART / nume)


# ───────────────────────── tabele ─────────────────────────
def _formate(df, zecimale):
    """Formatare pentru afișare: separator de mii, zecimale doar unde sunt necesare, „–” pentru lipsă."""
    fmt = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            valori = s.dropna()
            intreg = len(valori) > 0 and bool(np.allclose(valori, valori.round(0)))
            fmt[c] = "{:,.0f}" if intreg else "{:,.%df}" % zecimale
        else:
            fmt[c] = lambda v: "–" if pd.isna(v) else str(v)
    return fmt


def afiseaza_df(df, zecimale=3, inaltime=None):
    """Afișează un DataFrame formatat; ascunde indexul dacă e doar numerotare."""
    stil = df.style.format(_formate(df, zecimale), na_rep="–")
    optiuni = {"height": inaltime} if inaltime else {}
    st.dataframe(stil, width="stretch", hide_index=isinstance(df.index, pd.RangeIndex), **optiuni)


def descarca(df, nume_fisier, eticheta="⬇ Descarcă CSV"):
    st.download_button(eticheta, df.to_csv().encode("utf-8-sig"), file_name=nume_fisier, mime="text/csv", key=f"dl_{nume_fisier}_{id(df)}")


def arata_tabel(nume, index_col=0, zecimale=3, titlu=None, csv=False, inaltime=None):
    df = tabel(nume, index_col)
    if titlu:
        st.markdown(f"**{titlu}**")
    if df is None:
        st.warning(f"Lipsește `{nume}`. Rulează notebook-ul complet.")
        return None
    afiseaza_df(df, zecimale, inaltime)
    if csv:
        descarca(df, nume)
    return df


# ───────────────────────── figuri ─────────────────────────
def scara_imagini():
    """Procentul din lățime ocupat de figuri; vine din sliderul din bara laterală (app.py)."""
    return int(st.session_state.get("scara_img", 70))


def arata_fig(nume, legenda=None):
    """Figura din `artefacte/figuri/`, la scara aleasă în bara laterală. Cursorul peste figură arată butonul de mărire (ecran complet)."""
    cale = ART / "figuri" / f"{nume}.png"
    if not cale.exists():
        st.warning(f"Lipsește figura `{nume}.png`. Rulează notebook-ul complet.")
        return
    s = scara_imagini()
    if s >= 100:
        st.image(str(cale), caption=legenda, width="stretch")
        return
    try:
        c, _ = st.columns([s, 100 - s])
        c.image(str(cale), caption=legenda, width="stretch")
    except Exception:                       # coloane prea adânc imbricate: folosim lățime fixă în pixeli
        st.image(str(cale), caption=legenda, width=int(12 * s))


def scatter(df, x, y, color, ordine=None, culori=None, inaltime=460, punct_nou=None):
    """Grafic de puncte fără zoom cu rotița (nu se pierde poziția la derulare). `punct_nou` = (x, y, etichetă)."""
    scala = alt.Scale(domain=ordine, range=culori) if ordine and culori else alt.Undefined
    baza = alt.Chart(df).mark_circle(opacity=0.55, size=24).encode(
        x=alt.X(f"{x}:Q", scale=alt.Scale(zero=False)), y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color(f"{color}:N", scale=scala, legend=alt.Legend(title=None, orient="bottom")),
        tooltip=[color, alt.Tooltip(f"{x}:Q", format=".2f"), alt.Tooltip(f"{y}:Q", format=".2f")])
    grafic = baza
    if punct_nou is not None:
        nou = pd.DataFrame({x: [punct_nou[0]], y: [punct_nou[1]], "client": [punct_nou[2]]})
        grafic = baza + alt.Chart(nou).mark_point(shape="cross", size=300, color="black", strokeWidth=3).encode(
            x=f"{x}:Q", y=f"{y}:Q", tooltip=["client"])
    st.altair_chart(grafic.properties(height=inaltime), width="stretch")


# ───────────────────────── elemente de prezentare ─────────────────────────
def kpi(elemente):
    """Rând de cifre-cheie: listă de (etichetă, valoare[, explicație])."""
    cols = st.columns(len(elemente))
    for c, e in zip(cols, elemente):
        c.metric(e[0], e[1], help=e[2] if len(e) > 2 else None)


def observa(text, titlu="Ce observăm"):
    st.info(f"**{titlu}.** {text}", icon="🔎")


def concluzie(text):
    st.success(f"**Concluzie.** {text}", icon="✅")


def decizii(lista):
    """Tabel „decizie → de ce”: lista de perechi (decizie, motiv)."""
    rânduri = "\n".join(f"| {d} | {m} |" for d, m in lista)
    st.markdown(f"| Decizie | De ce |\n|---|---|\n{rânduri}")


def sarcina(intrebare, unitate, criterii, lista_decizii=None, nota=None):
    """Aspectul comun al taburilor „1 · Sarcina”: întrebare, unitate/țintă, criterii de succes, decizii."""
    c1, c2, c3 = st.columns([1.1, 1.1, 1.4])
    with c1.container(border=True):
        st.markdown("##### ❓ Întrebarea")
        st.markdown(intrebare)
    with c2.container(border=True):
        st.markdown("##### 🎯 Unitatea și ținta")
        st.markdown(unitate)
    with c3.container(border=True):
        st.markdown("##### ✅ Cum judecăm succesul")
        st.markdown("\n".join(f"{i}. {c}" for i, c in enumerate(criterii, 1)))
    if lista_decizii:
        st.markdown("##### Decizii de proiectare și motivele lor")
        decizii(lista_decizii)
    if nota:
        st.caption(nota)


def pentru_prezentare(bloc):
    """Expander „Pentru prezentare”: ce spui, de ce e făcut așa, întrebări posibile. `bloc` = dict din prezentare.py."""
    if not bloc or not st.session_state.get("note_prezentare", False):          # ascunse implicit; se activează din bara laterală
        return
    with st.expander("🎤 Pentru prezentare: ce spui și de ce"):
        if bloc.get("spui"):
            st.markdown(f"**Ce spui (20 de secunde).** {bloc['spui']}")
        if bloc.get("de_ce"):
            st.markdown("**De ce așa:**\n" + "\n".join(f"- {t}" for t in bloc["de_ce"]))
        for intrebare, raspuns in bloc.get("intrebari", []):
            st.markdown(f"**Dacă te întreabă:** *{intrebare}*  \n{raspuns}")


# ───────────────────────── formular comun Lab 1–2 ─────────────────────────
def construieste_x(v, features):
    """Aceleași transformări ca în notebook (secțiunea 3.3): log1p pe variabilele cu coadă lungă."""
    x = {
        "recenta_zile": v["recenta_zile"], "vechime_zile": v["vechime_zile"],
        "rata_retur": v["rata_retur"], "uk": int(v["uk"]),
        "log_frecventa": np.log1p(v["frecventa"]),
        "log_produse_distincte": np.log1p(v["produse_distincte"]),
        "log_vanzari_ultimele_90z": np.log1p(v["vanzari_ultimele_90z"]),
        "log_valoare_medie_factura": np.log1p(v["valoare_medie_factura"]),
        "log_retururi": np.log1p(abs(v["retururi_val"])),
    }
    return pd.DataFrame([x])[features]


def formular_client(cheie):
    """Formular cu variabilele brute ale unui client; valorile implicite sunt medianele clienților din date."""
    med = info()["mediane_clienti"]
    c1, c2, c3 = st.columns(3)
    v = {
        "recenta_zile": c1.number_input("Recență (zile de la ultima cumpărare)", 0, 700, int(med["recenta_zile"]), key=f"{cheie}_r"),
        "vechime_zile": c2.number_input("Vechime (zile de la prima cumpărare)", 0, 700, int(med["vechime_zile"]), key=f"{cheie}_v"),
        "frecventa": c3.number_input("Frecvență (nr. facturi)", 1, 500, max(1, int(med["frecventa"])), key=f"{cheie}_f"),
        "produse_distincte": c1.number_input("Produse distincte cumpărate", 1, 3000, max(1, int(med["produse_distincte"])), key=f"{cheie}_p"),
        "vanzari_ultimele_90z": c2.number_input("Vânzări în ultimele 90 de zile dinainte (£)", 0.0, 200000.0,
                                                float(med["vanzari_ultimele_90z"]), step=50.0, key=f"{cheie}_u"),
        "valoare_medie_factura": c3.number_input("Valoarea medie a unei facturi (£)", 1.0, 50000.0,
                                                 max(1.0, float(med["valoare_medie_factura"])), step=10.0, key=f"{cheie}_m"),
        "retururi_val": c1.number_input("Valoarea retururilor din istoric (£, negativă sau 0)", -100000.0, 0.0,
                                        float(med["retururi_val"]), step=10.0, key=f"{cheie}_rv"),
        "rata_retur": c2.number_input("Rata de retur (0–1)", 0.0, 1.0, float(med["rata_retur"]), step=0.01, key=f"{cheie}_rr"),
        "uk": c3.selectbox("Client din Marea Britanie?", [1, 0], format_func=lambda z: "Da" if z else "Nu", key=f"{cheie}_uk"),
    }
    return v
