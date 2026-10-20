import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (afiseaza_df, arata_fig, arata_tabel, cere_artefacte, info, kpi, observa, pentru_prezentare, tabel)

cere_artefacte("info_lab0.json", "laborator_0_datasetul.ipynb")
I = info("info_lab0.json")

st.title("Datasetul · Online Retail II")
st.caption(f"Tranzacțiile unui retailer online britanic (cadouri, mulți angrosiști), {I['data_min']} → {I['data_max']}. Aceleași date în toate laboratoarele.")
kpi([("Rânduri brute", f"{I['randuri_brute']:,}"), ("Facturi (vânzări curate)", f"{I['facturi_curate']:,}"), ("Produse", f"{I['produse']:,}"),
     ("Clienți cu ID", f"{I['clienti_id']:,}"), ("Țări", f"{I['tari']}")])

t1, t2, t3, t4, t5 = st.tabs(["Coloane", "Calitate și curățare", "În timp și pe țări", "Distribuții", "Istoric vs. viitor"])

with t1:
    st.subheader("Ce conține fiecare coloană")
    d = tabel("lab0_coloane.csv")
    if d is not None:
        st.table(d.style.format({"lipsă (%)": "{:.1f}", "valori distincte": "{:,.0f}"}))          # st.table: textul lung se împarte pe rânduri
    observa(f"Coloana `Customer ID` lipsește la {I['pct_fara_client']:.1f}% din rânduri (clienți neînregistrați): ei nu pot intra în analiza pe client (Laboratoarele 1–5), "
            "dar rămân în analiza pe factură și pe zi (Laboratorul 6).")
    pentru_prezentare({
        "spui": "Datele sunt linii de factură: ce produs, în ce cantitate, la ce preț, pentru ce client, în ce țară și când. O factură are mai multe linii.",
        "de_ce": ["Identificatorii (`Invoice`, `StockCode`, `Customer ID`) sunt citiți ca text: sunt etichete, nu cantități (ex.: `85123A`, `C581484`).",
                  "Un rând nu este un client: de aceea construim un tabel separat, cu un rând pe client (vezi „Istoric vs. viitor”)."],
        "intrebari": [("De ce lipsește Customer ID?", "Sunt clienți neînregistrați. Nu putem urmări un client fără identificator, deci îi excludem din analiza pe client, dar nu din cea pe factură.")]})

with t2:
    st.subheader("Probleme de calitate")
    arata_tabel("lab0_probleme.csv", zecimale=2)
    observa(f"Rândurile identice (duplicate exacte) sunt {I['pct_duplicate']:.1f}% din total. Nu le eliminăm: o linie repetată poate fi legitimă "
            "(același articol adăugat de două ori în comandă) și nu avem o coloană care să le distingă. Raportăm impactul, nu îl ascundem.",
            titlu="Despre duplicate")
    st.subheader("Curățare în straturi")
    arata_tabel("curatare.csv", index_col=None, zecimale=2)
    st.caption("Rândurile sunt mutate, nu șterse: servicii/ajustări, anulări, ajustări de stoc. Doar „vânzări curate” intră în cheltuiala clienților.")
    pentru_prezentare({
        "spui": f"Din {I['randuri_brute']:,} de rânduri, {I['randuri_curate']:,} sunt vânzări curate. Restul sunt servicii, anulări sau ajustări, păstrate în tabele separate.",
        "de_ce": ["Documentația UCI spune că `StockCode` are 5 cifre și că doar facturile „C” sunt anulări; datele conțin și coduri de servicii și facturi „A”. Am verificat în date.",
                  "Un rând nu poate fi în două straturi (verificat în cod cu `assert`).",
                  "Anulările scad din cheltuiala netă, iar serviciile și ajustările sunt studiate în Laboratorul 6."],
        "intrebari": [("De ce n-ați șters pur și simplu rândurile problematice?", "Fiecare strat are un rol în alt laborator; ștergerea ar pierde informație și ar ascunde cât de mult conta.")]})

with t3:
    lunar = tabel("lab0_lunar.csv", index_col=None)
    tari = tabel("lab0_tari.csv", index_col=None)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Venitul lunar din vânzări")
        if lunar is not None:
            st.altair_chart(alt.Chart(lunar).mark_bar().encode(
                x=alt.X("luna:N", title=None, sort=None), y=alt.Y("venit vânzări (£):Q", title="£"),
                tooltip=["luna", alt.Tooltip("venit vânzări (£):Q", format=",.0f"), alt.Tooltip("retururi (£):Q", format=",.0f")]).properties(height=300), width="stretch")
    with c2:
        st.subheader("Primele 10 țări")
        if tari is not None:
            top = tari[tari["țara"] != "Alte țări"]
            st.altair_chart(alt.Chart(top).mark_bar().encode(
                y=alt.Y("țara:N", sort="-x", title=None), x=alt.X("% din venit:Q", title="% din venit"),
                tooltip=["țara", alt.Tooltip("% din venit:Q", format=".1f")]).properties(height=300), width="stretch")
    observa(f"Luna cu cel mai mare venit este {I['luna_maxima']} ({I['pct_venit_luna_maxima']:.1f}% din venitul total). Marea Britanie aduce {I['pct_venit_uk']:.0f}% din venit. "
            "Vânzările cresc spre toamnă, înainte de sărbători, deci fereastra de 90 de zile de la final nu este una „tipică”.")
    if lunar is not None:
        afiseaza_df(lunar.set_index("luna"), zecimale=0, inaltime=260)
    if tari is not None:
        afiseaza_df(tari.set_index("țara"), zecimale=1)
    pentru_prezentare({
        "spui": f"Vânzările au un sezon clar (vârf la {I['luna_maxima']}) și sunt concentrate în Marea Britanie ({I['pct_venit_uk']:.0f}% din venit).",
        "de_ce": ["Sezonalitatea explică de ce rezultatele depind de fereastra aleasă: o singură dată de tăiere e o limită declarată.",
                  "Concentrarea pe o țară explică de ce `uk` apare ca variabilă în modele."],
        "intrebari": [("Ce limită are asta?", "Cu ~2 ani de date avem o singură sărbătoare de iarnă completă pe an, deci sezonalitatea anuală nu se poate separa complet de efecte singulare.")]})

with t4:
    st.subheader("Cum arată o variabilă?")
    sursa = st.radio("Nivelul datelor", ["Linii de factură", "Clienți (variabile din istoric)"], horizontal=True)
    df = tabel("lab0_esantion_linii.csv", index_col=None) if sursa.startswith("Linii") else tabel("lab0_clienti_variabile.csv", index_col=None)
    if df is not None:
        c1, c2, c3 = st.columns([2, 1, 1])
        var = c1.selectbox("Variabila", list(df.columns))
        logaritm = c2.checkbox("Scară logaritmică (log1p)", value=True)
        fara_coada = c3.checkbox("Doar până la percentila 99", value=False)
        s = df[var].dropna()
        if fara_coada:
            s = s[s <= s.quantile(0.99)]
        x = np.log1p(s.clip(lower=0)) if logaritm else s
        h = pd.DataFrame({"valoare": x})
        st.altair_chart(alt.Chart(h).mark_bar().encode(
            x=alt.X("valoare:Q", bin=alt.Bin(maxbins=50), title=f"log1p({var})" if logaritm else var), y=alt.Y("count()", title="număr"),
        ).properties(height=300), width="stretch")
        rez = pd.DataFrame({"valoare": {"număr": len(s), "medie": s.mean(), "mediană": s.median(), "maxim": s.max(), "asimetrie": s.skew()}})
        c1, c2 = st.columns([1, 2])
        with c1:
            afiseaza_df(rez, zecimale=2)
        with c2:
            observa("Asimetria > 1 înseamnă coadă lungă (puține valori foarte mari). De aceea aplicăm `log1p` în laboratoare: altfel metodele pe distanță "
                    "și erorile pătratice ar fi dominate de câțiva angrosiști. Bifează „Scară logaritmică” ca să vezi diferența.")
    forma = tabel("lab0_forma_clienti.csv")
    if forma is not None:
        arata_tabel("lab0_forma_clienti.csv", zecimale=2, titlu="Forma distribuțiilor, pe clienți")
    pentru_prezentare({
        "spui": "Aproape toate variabilele au coadă lungă: mulți clienți mici și puțini foarte mari. Asta ne-a determinat să folosim `log1p` în laboratoare.",
        "de_ce": ["Media e trasă de extreme; mediana descrie mai bine clientul tipic.", "Angrosiștii nu sunt erori: sunt clienți reali, deci nu-i ștergem."],
        "intrebari": [("De ce log1p și nu log?", "`log1p(x) = log(1 + x)` este definit și pentru zero, care apare des (frecvență, retururi, vânzări).")]})

with t5:
    st.subheader("Separarea istoricului de viitor")
    arata_fig("lab0_axa_timp")
    arata_tabel("lab0_separare.csv", zecimale=0)
    observa(f"Variabilele (recență, frecvență, valoare medie, retururi…) se calculează doar din istoric, până la {I['data_taiere']}; ținta (a cumpărat? cât a cheltuit?) doar din "
            f"ultimele {I['zile_fereastra']} de zile. Dacă o variabilă ar conține informație din viitor, modelul ar părea excelent în laborator și ar eșua în realitate (scurgere de informație).",
            titlu="De ce data de tăiere")
    pentru_prezentare({
        "spui": f"Toate modelele pornesc din aceeași dată de tăiere, {I['data_taiere']}: ce se știa până atunci este intrarea, ce s-a întâmplat după este ținta.",
        "de_ce": ["Imită folosirea reală: la o anumită dată nu știm încă viitorul.", "Aceeași dată în toate laboratoarele face rezultatele comparabile."],
        "intrebari": [("Ce e scurgerea de informație (data leakage)?", "Când o variabilă conține, direct sau indirect, răspunsul. Modelul pare bun la antrenare și greșește în producție."),
                      ("Rezultatele depind de data aleasă?", "Da, o singură dată de tăiere este o limită; ideal s-ar repeta pe mai multe date.")]})
