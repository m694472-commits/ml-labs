import numpy as np
import pandas as pd
import streamlit as st

from prezentare import lab
from utils import (afiseaza_df, arata_fig, arata_tabel, cere_artefacte, concluzie, info, kpi, model, observa, pentru_prezentare, sarcina,
                   scatter, tabel, tabel_text)

cere_artefacte("info_lab4.json", "laborator_4_reducerea_dimensionalitatii.ipynb")
I = info("info_lab4.json")
P = lab(4)
ORDINE = [I["clase"][k] for k in sorted(I["clase"])]
CULORI = ["#7f7f7f", "#1f77b4", "#d62728"]

st.title("Laborator 4 · Reducerea dimensionalității")
st.caption(P["intrebare"])
kpi(P["kpi"])

t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Sarcina", "2 · Datele", "3 · Pregătirea", "4 · Modelarea", "5 · Evaluarea", "6 · Implementarea"])

with t1:
    sarcina(
        intrebare="Putem rezuma **ce cumpără** fiecare client (sute de produse) în câteva dimensiuni, fără să pierdem informația utilă?",
        unitate=(f"**Unitatea:** clientul · matrice client × produs.\n\n**Dimensiuni:** {I['dimensiuni']:,} produse (unități cumpărate în istoric).\n\n"
                 "**Etichete (doar pentru LDA și evaluare):** comportamentul din ultimele 90 de zile, în 3 clase."),
        criterii=["**PCA:** câte componente sunt necesare; comparăm regula de 95% din varianță cu alegerea prin validare încrucișată.",
                  "**LDA:** separare între clase **pe test**, nu doar pe train.", "**t-SNE:** păstrarea vecinătăților (trustworthiness) și structură vizibilă în 2D."],
        lista_decizii=[
            ("PCA, LDA și t-SNE împreună", "Sunt trei logici diferite: liniar nesupervizat, liniar supervizat, neliniar pentru vizualizare."),
            ("Blestemul dimensionalității verificat pe date", "Cu multe dimensiuni distanțele se uniformizează; măsurăm asta în loc să o presupunem."),
            ("k ales pe performanță, nu doar pe varianță", "Regula de 95% nu ține cont de ce facem cu datele; validarea încrucișată pe train o compară cu o alegere bazată pe AUC."),
            ("t-SNE doar pentru vizualizare", "Nu poate proiecta clienți noi; îl rulăm pe 50 de componente PCA pentru viteză și mai puțin zgomot."),
        ])
    pentru_prezentare(P["taburi"][1])

with t2:
    kpi([("Clienți (rânduri)", f"{I['clienti']:,}"), ("Produse păstrate (D)", f"{I['dimensiuni']:,}"), ("Prag de păstrare", f"≥ {I['prag_produs']:.0%} din clienți"),
         ("Mediana cheltuielii (cumpărători)", f"{I['mediana_cheltuiala']:,.0f} £")])
    st.subheader("Matricea client × produs")
    arata_tabel("lab4_matrice.csv", titlu=None)
    st.caption(f"Păstrăm produsele cumpărate de cel puțin {I['prag_produs']:.0%} dintre clienți; tabelul arată ce acoperire a istoricului rămâne.")
    st.subheader("Clasele de comportament viitor (pentru LDA și evaluare)")
    arata_tabel("lab4_clase.csv", zecimale=1)
    observa(f"0 = nu cumpără; 1 = cheltuială netă ≤ mediana cumpărătorilor ({I['mediana_cheltuiala']:,.0f} £); 2 = peste mediană. "
            "Mediana e calculată doar pe cumpărători, ca clasele 1 și 2 să fie egale ca mărime.")
    pentru_prezentare(P["taburi"][2])

with t3:
    st.markdown("**Împărțire 80/20 stratificată înainte de orice scalare sau PCA** (scalerul și componentele se învață doar pe train), "
                "apoi **standardizare z-score** (PCA e sensibil la scara variabilelor).")
    st.subheader("Blestemul dimensionalității: distanțele se uniformizează")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_fig("lab4_blestem")
    with c2:
        arata_tabel("lab4_blestem.csv")
    observa("Coeficientul de variație = abatere standard / medie a distanțelor dintre perechi de clienți. Când scade, toate distanțele devin asemănătoare "
            "și „cel mai apropiat vecin” nu mai spune mult.")
    pentru_prezentare(P["taburi"][3])

with t4:
    st.subheader("PCA")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_tabel("lab4_verificare_pca.csv", zecimale=3, titlu="Valorile proprii „de mână” (numpy) vs. sklearn")
    with c2:
        arata_tabel("lab4_pca_praguri.csv", zecimale=1, titlu="Componente necesare pentru fiecare prag de varianță")
    arata_fig("lab4_pca_scree")
    arata_tabel("lab4_incarcari.csv", titlu="Încărcările primelor două componente (primele 200 de rânduri)", inaltime=300)
    st.subheader("Câte componente? Regula de 95% vs. validarea încrucișată")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_fig("lab4_alegere_k")
    with c2:
        arata_tabel("lab4_cv_k.csv", zecimale=4)
    observa(f"Regula de 95% cere {I['k95']} componente; validarea încrucișată (3-fold, doar pe train) alege {I['k_cv']}. "
            "Dincolo de acest punct, componentele suplimentare adaugă zgomot și modelul supraantrenează.")
    st.markdown("**Performanța unui model în funcție de numărul de dimensiuni** (AUC macro one-vs-rest, 3 clase)")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_fig("lab4_performanta_dimensiuni")
    with c2:
        p = tabel("lab4_performanta_dimensiuni.csv")
        if p is not None:
            afiseaza_df(p.drop(columns="k"), zecimale=3)
    st.subheader("LDA")
    arata_tabel("lab4_lda.csv")
    arata_fig("lab4_lda")
    st.subheader("t-SNE")
    arata_tabel("lab4_tsne.csv")
    arata_fig("lab4_tsne")
    pentru_prezentare(P["taburi"][4])

with t5:
    st.subheader("Comparația metodelor")
    arata_tabel("lab4_comparatie.csv", zecimale=3, csv=True)
    st.success(f"PCA: {I['k_cv']} componente alese prin validare (regula de 95%: {I['k95']}, din {I['dimensiuni']}) · "
               f"LDA (după PCA {I['k_cv']}): acuratețe test {I['lda_acuratete_test']:.3f} față de baza {I['baza_acuratete']:.3f} · "
               f"t-SNE: trustworthiness {I['trust_tsne']:.3f} față de {I['trust_pca2d']:.3f} la PCA 2D")
    arata_fig("lab4_comparatie_2d")
    st.subheader("Explorare interactivă")
    proiectii = tabel("lab4_proiectii.csv")
    varianta = tabel("lab4_pca_varianta.csv", index_col=None)
    c1, c2 = st.columns([2, 1])
    with c1:
        metoda = st.radio("Metoda", ["PCA", "LDA", "t-SNE"], horizontal=True)
        col_x, col_y = {"PCA": ("PC1", "PC2"), "LDA": ("LD1", "LD2"), "t-SNE": ("tSNE1", "tSNE2")}[metoda]
        multime = st.multiselect("Mulțime", ["train", "test"], default=["train", "test"])
        if proiectii is not None:
            scatter(proiectii[proiectii["set"].isin(multime)], col_x, col_y, "clasa", ORDINE, CULORI)
    with c2:
        if varianta is not None:
            k = st.slider("Număr de componente PCA", 1, len(varianta), min(10, len(varianta)))
            st.metric("Varianță explicată cumulat", f"{varianta.loc[k - 1, 'cumulat']:.1%}")
            st.caption("Mută cursorul ca să vezi cât din informație rămâne cu mai puține componente.")
    pentru_prezentare(P["taburi"][5])

with t6:
    st.subheader("Unde cade un client nou?")
    st.caption("Alegi produsele cumpărate; clientul este proiectat în spațiul PCA și LDA, față de ceilalți clienți.")
    produse = tabel_text("lab4_produse.csv")
    if produse is not None:
        produse["eticheta"] = produse["StockCode"] + " · " + produse["Description"]
        with st.form("form_dim"):
            alese = st.multiselect("Produse cumpărate de client", produse["eticheta"].tolist(), placeholder="Alege unul sau mai multe produse")
            cant = st.number_input("Unități cumpărate din fiecare produs", 1, 1000, 5)
            trimis = st.form_submit_button("Proiectează")
        if trimis and alese:
            x = pd.DataFrame(0.0, index=[0], columns=produse["StockCode"].tolist())
            for e in alese:
                x[e.split(" · ")[0]] = np.log1p(cant)
            pc = model("model_pca.joblib").transform(x)[0, :2]
            ld = model("model_lda.joblib").transform(x)[0, :2]
            a, b, c, d = st.columns(4)
            a.metric("PC1", f"{pc[0]:.2f}")
            b.metric("PC2", f"{pc[1]:.2f}")
            c.metric("LD1", f"{ld[0]:.2f}")
            d.metric("LD2", f"{ld[1]:.2f}")
            proiectii = tabel("lab4_proiectii.csv")
            if proiectii is not None:
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("**Spațiul PCA**")
                    scatter(proiectii, "PC1", "PC2", "clasa", ORDINE, CULORI, inaltime=380, punct_nou=(pc[0], pc[1], "client nou"))
                with g2:
                    st.markdown("**Spațiul LDA**")
                    scatter(proiectii, "LD1", "LD2", "clasa", ORDINE, CULORI, inaltime=380, punct_nou=(ld[0], ld[1], "client nou"))
                st.caption("Crucea neagră este clientul nou. Culorile arată clasa de comportament a clienților existenți; nu estimăm clasa clientului nou.")
        elif trimis:
            st.warning("Alege cel puțin un produs.")
    pentru_prezentare(P["taburi"][6])

concluzie(P["concluzie"])
