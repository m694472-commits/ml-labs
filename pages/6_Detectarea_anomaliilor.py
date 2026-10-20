import numpy as np
import pandas as pd
import streamlit as st

from prezentare import lab
from utils import (afiseaza_df, arata_fig, arata_tabel, cere_artefacte, concluzie, info, kpi, model, observa, pentru_prezentare, sarcina, tabel)

cere_artefacte("info_lab6.json", "laborator_6_detectarea_anomaliilor.ipynb")
I = info("info_lab6.json")
P = lab(6)

st.title("Laborator 6 · Detectarea anomaliilor")
st.caption(P["intrebare"])
kpi(P["kpi"])

t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Sarcina", "2 · Datele", "3 · Pregătirea", "4 · Modelarea", "5 · Evaluarea", "6 · Implementarea"])

with t1:
    sarcina(
        intrebare="Ce **facturi**, **clienți** și **zile** se abat atât de mult de la normal încât merită verificate?",
        unitate=(f"**Unități:** **{I['facturi']:,}** facturi, **{I['clienti']:,}** clienți, zilele din serie.\n\n"
                 f"**Buget de investigare:** {I['contaminare']:.0%} din observații.\n\n"
                 "**Tipuri:** punctuale (univariate și multivariate), contextuale (față de țară / față de istoricul clientului), colective (zile neobișnuite)."),
        criterii=["**Acord** între metode independente și stabilitate la altă inițializare.", "**Explicații**: pentru fiecare anomalie, ce o face neobișnuită.",
                  "**Cazul cunoscut** (comanda de 80.995 de bucăți) trebuie găsit.", "**Contextul** arată anomalii pe care analiza globală nu le vede."],
        lista_decizii=[
            ("Anomaliile nu se șterg", "Pot fi erori, dar și cauze naturale (un client angrosist): le investigăm."),
            ("Fără acuratețe", "Nu avem etichete; încrederea vine din acord între metode, stabilitate, explicații și un caz cunoscut."),
            ("Șase metode, același buget", "Metodele văd lucruri diferite (izolare, densitate, distanță); un punct marcat de mai multe e mai sigur o anomalie."),
            ("Context și serie de timp", "O factură poate fi normală global și anormală pentru clientul ei; o zi poate fi anormală fără nicio factură anormală."),
        ])
    pentru_prezentare(P["taburi"][1])

with t2:
    kpi([("Facturi", f"{I['facturi']:,}"), ("Clienți", f"{I['clienti']:,}"), ("Zile în serie", f"{I['zile_evaluate']:,}"),
         ("Buget de investigare", f"{I['contaminare']:.0%}")])
    st.subheader("Metode grafice și statistice")
    arata_fig("lab6_grafice_valoare")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_tabel("lab6_univariat.csv", zecimale=3, titlu="IQR și Z-score pe valoarea facturii")
    with c2:
        observa("Pe date asimetrice, Z-score se „maschează”: media și abaterea standard sunt trase de extreme. IQR e mai robust, iar `log1p` schimbă mult rezultatul.")
    arata_tabel("lab6_cele_mai_mari.csv", titlu="Cele mai mari 8 facturi (reale sau erori?)", zecimale=2)
    with st.expander("Pairs plot (facturi)"):
        arata_fig("lab6_pairs")
    arata_tabel("curatare.csv", index_col=None, zecimale=2, titlu="Curățarea datelor (comună laboratoarelor)")
    pentru_prezentare(P["taburi"][2])

with t3:
    st.subheader("Cum tratăm valorile extreme?")
    arata_tabel("lab6_tratari.csv", zecimale=2)
    st.info("Pentru detecție **nu tratăm** extremele, pentru că ele sunt ținta. Tratarea (winsorizare, `log1p`) este pasul potrivit înainte de un model predictiv. "
            "Singura transformare folosită aici este `log1p`, urmată de standardizare.")
    pentru_prezentare(P["taburi"][3])

with t4:
    st.subheader("Punctuale multivariate: șase metode, același buget")
    c1, c2 = st.columns(2)
    with c1:
        arata_tabel("lab6_consens.csv", zecimale=0, titlu="Câte metode marchează o observație")
    with c2:
        observa("DBSCAN nu primește un buget: își stabilește singur zgomotul (ε din graficul k-distanță). "
                "O observație marcată de multe metode este mai sigur o anomalie decât una marcată doar de una.")
    st.subheader("Contextuale")
    arata_tabel("lab6_contextuale.csv", zecimale=2)
    ctx = tabel("lab6_contextuale_facturi.csv", index_col=None)
    if ctx is not None:
        st.markdown("**Facturi anormale doar în contextul clientului (nevăzute de analiza globală)**")
        cols = [c for c in ["Invoice", "client", "produs_principal", "nr_linii", "cantitate_totala", "valoare_totala", "z_global", "z_client"] if c in ctx.columns]
        afiseaza_df(ctx[cols].head(100), zecimale=2, inaltime=300)
    st.subheader("Colective: serie de timp cu variabile de calendar")
    arata_tabel("lab6_arima.csv", zecimale=1, titlu="Alegerea modelului după AIC (cu și fără calendar)")
    observa(f"Fără variabile de calendar sunt {I['zile_anormale_fara_calendar']} zile anormale; cu calendar (sărbători UK + perioada de iarnă), {I['zile_anormale']}. "
            "O parte din „anomaliile” fără calendar sunt abateri previzibile (zile libere), nu evenimente de investigat.")
    arata_fig("lab6_serie_zilnica")
    arata_tabel("lab6_episoade.csv", index_col=None, zecimale=0, titlu="Episoade (zile anormale apropiate, de același sens)")
    pentru_prezentare(P["taburi"][4])

with t5:
    st.subheader("Acordul dintre metode")
    c1, c2 = st.columns([2, 1])
    with c1:
        arata_fig("lab6_jaccard")
    with c2:
        st.metric("Stabilitate Isolation Forest (5 inițializări)", f"{I['stabilitate_if']:.3f}")
        st.caption("Jaccard = intersecție / reuniune; 1 = aceleași puncte. Valoare mică între două metode înseamnă că „văd” alte tipuri de anomalii.")
    st.subheader("Cazul cunoscut")
    caz = tabel("lab6_caz_cunoscut.csv", index_col=None)
    if caz is not None:
        afiseaza_df(caz, zecimale=1)
        observa(f"Metode care au marcat-o: {', '.join(I['caz_metode'])}.", titlu=f"Comanda de 80.995 de bucăți (factura {I['caz_id']})")
    else:
        st.caption("Factura cu 80.995 de bucăți nu există în acest set de date.")
    st.subheader("Anomalii de consens și explicația lor")
    st.caption(f"Consens = marcată de cel puțin {I['prag_consens']} din {len(I['metode'])} metode.")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_tabel("lab6_importanta.csv", zecimale=2, titlu="Cât contează pentru business")
    with c2:
        arata_tabel("lab6_motive.csv", zecimale=1, titlu="Motivul principal")
    fa = tabel("lab6_anomalii_facturi.csv", index_col=None)
    if fa is not None:
        n = st.slider("Câte facturi arăt", 5, min(300, len(fa)), min(25, len(fa)))
        min_m = st.slider("Număr minim de metode", I["prag_consens"], len(I["metode"]), I["prag_consens"])
        afiseaza_df(fa[fa["nr_metode"] >= min_m].head(n), zecimale=2, inaltime=380)
    ca = tabel("lab6_anomalii_clienti.csv", index_col=None)
    if ca is not None:
        st.markdown("**Clienți anormali (consens)**")
        afiseaza_df(ca.head(50), zecimale=2, inaltime=300)
    st.subheader("Rezumat pe tipuri")
    arata_tabel("lab6_rezumat.csv", zecimale=0, csv=True)
    pentru_prezentare(P["taburi"][5])

with t6:
    st.subheader("Este neobișnuită această factură?")
    st.caption("Modelul: scalare + Isolation Forest, antrenat pe toate facturile. Prețul mediu se calculează din valoare și cantitate.")
    with st.form("form_anomalie"):
        c1, c2, c3 = st.columns(3)
        nr = c1.number_input("Număr de linii", 1, 500, 12)
        cant = c2.number_input("Cantitate totală (bucăți)", 1, 200000, 150)
        val = c3.number_input("Valoare totală (£)", 0.5, 500000.0, 350.0, step=10.0)
        trimis = st.form_submit_button("Verifică")
    if trimis:
        x = pd.DataFrame({"nr_linii": [nr], "cantitate_totala": [cant], "valoare_totala": [val], "pret_mediu": [val / cant]})
        x = np.log1p(x)
        x.columns = I["variabile_factura"]
        m = model("model_anomalii.joblib")
        scor, anormal = float(m.decision_function(x)[0]), bool(m.predict(x)[0] == -1)
        z = {c: (x.iloc[0][c] - I["mediana_log"][c]) / I["mad_log"][c] for c in x.columns}
        a, b = st.columns(2)
        a.metric("Verdict", "ANOMALIE" if anormal else "Normală")
        b.metric("Scor Isolation Forest", f"{scor:.3f}", help="Sub 0 = anormală; cu cât mai mic, cu atât mai neobișnuită.")
        expl = pd.DataFrame({"variabilă": [I["nume_variabile"][c] for c in z], "Z robust": list(z.values())})
        expl["sens"] = np.where(expl["Z robust"] > 0, "mai mare decât tipic", "mai mică decât tipic")
        afiseaza_df(expl.sort_values("Z robust", key=np.abs, ascending=False), zecimale=2)
        st.caption("Z robust = (x − mediana) / (1,4826 · MAD), pe scara log1p. Valori peste ±3 indică o abatere puternică. Verdictul este un motiv de verificare, nu o sentință.")
    pentru_prezentare(P["taburi"][6])

concluzie(P["concluzie"])
