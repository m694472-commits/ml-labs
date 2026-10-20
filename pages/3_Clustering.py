import numpy as np
import pandas as pd
import streamlit as st

from prezentare import lab
from utils import (afiseaza_df, arata_fig, arata_tabel, cere_artefacte, concluzie, info, kpi, model, observa, pentru_prezentare, sarcina, tabel)

cere_artefacte("info_lab3.json", "laborator_3_clustering.ipynb")
I = info("info_lab3.json")
P = lab(3)

st.title("Laborator 3 · Clustering")
st.caption(P["intrebare"] + " (învățare nesupervizată)")
kpi(P["kpi"])

t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Sarcina", "2 · Datele", "3 · Pregătirea", "4 · Modelarea", "5 · Evaluarea", "6 · Implementarea"])

with t1:
    sarcina(
        intrebare="Ce **grupuri naturale de clienți** există, după cum cumpără? Nu avem nicio țintă (învățare nesupervizată).",
        unitate=(f"**Unitatea:** clientul (**{I['clienti']:,}** clienți cu istoric).\n\n"
                 "**Variabilele:** R (recență), F (frecvență), V (valoarea medie a unei facturi).\n\n"
                 "**Aplicația:** segmentarea clienților pentru marketing direcționat."),
        criterii=["**Separare internă:** silhouette mare, Davies-Bouldin mic, Calinski-Harabasz mare.",
                  "**Interpretabilitate:** fiecare segment se descrie în cuvinte și are o dimensiune utilă.",
                  "**Utilitate:** segmentele diferă în comportamentul *viitor* (nevăzut de algoritm)."],
        lista_decizii=[
            ("RFM în loc de toate variabilele", "Trei variabile pe care un om de marketing le înțelege imediat: cât de recent, cât de des, cât valorează."),
            ("M (vânzări totale) exclus", "M este exact F × V: repetă informația și ar da dublă pondere clienților mari."),
            ("`log1p` și standardizare", "Distanța euclidiană ar fi dominată de coada lungă și de variabila cu scara cea mai mare."),
            ("Evaluare și pe viitor", "Fără țintă, silhouette singur nu arată dacă segmentele ajută; verificăm și ce fac în ultimele 90 de zile."),
        ])
    pentru_prezentare(P["taburi"][1])

with t2:
    kpi([("Clienți", f"{I['clienti']:,}"), ("Variabile folosite", str(len(I["variabile_x"]))),
         ("Transformate cu log1p", ", ".join(I["log_pe"])), ("Variabilă exclusă", "M (= F × V)")])
    st.subheader("Variabile candidate: R, F, M, V")
    st.markdown("**R**ecență, **F**recvență, **M**onetar (vânzări totale), **V** = valoarea medie a unei facturi. "
                "Aplicăm `log1p` celor cu asimetrie > 1, pentru că distanța euclidiană ar fi dominată de coada lungă.")
    arata_tabel("lab3_asimetrie.csv", titlu="Asimetria înainte și după transformare")
    arata_fig("lab3_distributii", "Distribuții brute (sus) și după log1p (jos).")
    arata_tabel("lab3_corelatie_candidati.csv", titlu="Corelațiile dintre candidate (după transformare)")
    observa("M total este exact F × V, deci repetă informația lor; folosim R, F, V.")
    pentru_prezentare(P["taburi"][2])

with t3:
    st.subheader("Date finale pentru clustering")
    st.markdown(f"Variabile: `{', '.join(I['variabile_x'])}` · **{I['clienti']:,}** clienți · scalare cu `StandardScaler`.")
    arata_tabel("lab3_extreme.csv", titlu="Extreme (Z-score și IQR)")
    st.caption("Nu ștergem clienții cu valori mari: sunt clienții valoroși pe care segmentarea trebuie să-i găsească.")
    arata_tabel("curatare.csv", index_col=None, zecimale=2, titlu="Curățarea datelor (comună laboratoarelor)")
    pentru_prezentare(P["taburi"][3])

with t4:
    st.subheader("Câte clustere? Cotul și silhouette")
    arata_fig("lab3_alegere_k")
    arata_tabel("lab3_alegere_k.csv")
    st.caption(f"Regulă: silhouette maxim pentru K între 3 și 8 → **K = {I['k_bun_silhouette']}**.")
    st.subheader("Clustering ierarhic: tipuri de legătură")
    arata_tabel("lab3_ierarhic.csv", zecimale=2)
    st.caption("`cel mai mare cluster %` mare = efect de lanț (un cluster uriaș, restul minuscule): silhouette bun, dar segmente inutile.")
    arata_fig("lab3_dendrograma")
    st.subheader("DBSCAN")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_fig("lab3_dbscan_kdist")
    with c2:
        arata_tabel("lab3_dbscan.csv", titlu=f"Sensibilitate la ε (min_samples = {I['min_samples_dbscan']})")
    st.subheader("Gaussian Mixture Models (clustering soft)")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_fig("lab3_gmm_bic")
    with c2:
        g = tabel("lab3_gmm_bic.csv", index_col=None)
        if g is not None:
            st.markdown("**BIC pe forma covarianței (mai mic = mai bine)**")
            afiseaza_df(g.pivot(index="K", columns="covarianță", values="BIC"), zecimale=1)
            if "varianța minimă" in g.columns:
                st.markdown("**Varianța minimă a componentelor** (aproape de valoarea de regularizare = componentă degenerată)")
                afiseaza_df(g.pivot(index="K", columns="covarianță", values="varianța minimă"), zecimale=3)
    pentru_prezentare(P["taburi"][4])

with t5:
    st.subheader("Comparația algoritmilor")
    arata_tabel("lab3_comparatie.csv", csv=True)
    st.success(f"Model final: **{I['model']}** · silhouette = {I['silhouette']:.3f} · Davies-Bouldin = {I['davies_bouldin']:.3f} · "
               f"Calinski-Harabasz = {I['calinski_harabasz']:,.0f} · V-measure = {I['v_measure']:.3f}")
    st.caption("Regulă: dintre algoritmii care pot segmenta clienți noi (K-means, GMM) alegem silhouette-ul maxim. "
               "Metricile externe folosesc `a_cumparat` (viitorul), nevăzut de algoritm.")
    st.subheader("Profilul segmentelor")
    arata_tabel("lab3_profil_segmente.csv", zecimale=1, csv=True)
    st.caption("Numele vin din reguli, nu sunt alese manual: fiecare variabilă (mediana segmentului) se compară cu terțilele clienților → mic / mediu / mare.")
    observa(f"Segmentul de top reprezintă {I['pct_clienti_top_segment']:.0f}% din clienți și {I['pct_vanzari_top_segment']:.0f}% din vânzări. "
            "Datele formează un nor continuu, nu grupuri separate clar: segmentele sunt o împărțire utilă, nu o descoperire a unor clase naturale.")
    arata_fig("lab3_proiectii")
    arata_fig("lab3_centroizi")
    pentru_prezentare(P["taburi"][5])

with t6:
    st.subheader("În ce segment intră un client?")
    med = I["mediane"]
    with st.form("form_clust"):
        c1, c2, c3 = st.columns(3)
        r = c1.number_input("Recență (zile de la ultima cumpărare)", 0, 700, int(med[I["brut"]["R"]]))
        f = c2.number_input("Frecvență (nr. facturi)", 1, 500, max(1, int(med[I["brut"]["F"]])))
        v = c3.number_input("Valoarea medie a unei facturi (£)", 1.0, 50000.0, max(1.0, float(med[I["brut"]["V"]])), step=10.0)
        trimis = st.form_submit_button("Segmentează")
    if trimis:
        brut = {"R": r, "F": f, "V": v}
        x = pd.DataFrame([{("log_" if k in I["log_pe"] else "") + I["brut"][k]: (np.log1p(brut[k]) if k in I["log_pe"] else brut[k])
                           for k in ["R", "F", "V"]}])[I["variabile_x"]]
        seg = int(model("model_clustering.joblib").predict(x)[0])
        st.metric("Segment", f"{seg} · {I['nume_segmente'][str(seg)]}")
        prof = tabel("lab3_profil_segmente.csv")
        if prof is not None and seg in prof.index:
            st.markdown("**Cum arată segmentul în date**")
            afiseaza_df(prof.loc[[seg]], zecimale=1)
    pentru_prezentare(P["taburi"][6])

concluzie(P["concluzie"])
