import pandas as pd
import streamlit as st

from prezentare import lab
from utils import (afiseaza_df, arata_fig, arata_tabel, cere_artefacte, concluzie, info, kpi, observa, pentru_prezentare, sarcina, tabel, tabel_text)

cere_artefacte("info_lab5.json", "laborator_5_analiza_tiparelor.ipynb")
I = info("info_lab5.json")
P = lab(5)

st.title("Laborator 5 · Analiza și explorarea tiparelor")
st.caption(P["intrebare"])
kpi(P["kpi"])

NUM = ["support", "confidence", "lift", "support_test", "confidence_test", "lift_test"]
reguli = tabel_text("lab5_reguli.csv")
if reguli is not None:
    reguli[NUM] = reguli[NUM].astype(float)

t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Sarcina", "2 · Datele", "3 · Pregătirea", "4 · Modelarea", "5 · Evaluarea", "6 · Implementarea"])

with t1:
    sarcina(
        intrebare="Ce produse se **cumpără împreună** și ce reguli „dacă A, atunci B” putem folosi pentru recomandări, plasarea produselor și pachete?",
        unitate=(f"**Unitatea:** factura (un coș = o factură).\n\n**Perioade:** tipare căutate pe {I['coșuri_train']:,} coșuri dinainte de {I['data_taiere']}, "
                 f"verificate pe {I['coșuri_test']:,} coșuri de după.\n\n"
                 "**Metrici:** *support* = cât de des apare; *confidence* = P(B | A); *lift* = confidence / support(B) (1 = independență, > 1 = asociere pozitivă)."),
        criterii=["Apriori, FP-Growth și ECLAT dau **aceleași** seturi frecvente (la timpi diferiți).",
                  "Regulile **rezistă în timp**: găsite pe istoric, verificate pe perioada următoare.",
                  "Ca recomandări, regulile **bat baza** (produsele populare) la testul „ascunde un produs din coș și ghicește-l”."],
        lista_decizii=[
            ("Factura ca unitate", "Un coș cumpărat odată reflectă produsele alese împreună; clientul pe tot istoricul ar amesteca perioade."),
            ("Lift, nu doar confidence", "Un produs popular are confidence mare cu orice; lift împarte la popularitatea lui și arată dacă asocierea e reală."),
            ("Validare în timp", "O regulă testată pe aceleași date doar le memorează; testul pe perioada următoare arată dacă rămâne valabilă."),
            ("Praguri alese pe criteriu", f"Suportul minim ({I['suport_minim']}) e cel mai mic pentru care rămân cel mult 1.000 de reguli verificabile de un analist."),
        ])
    pentru_prezentare(P["taburi"][1])

with t2:
    kpi([("Coșuri în analiză", f"{I['coșuri_train'] + I['coșuri_test']:,}"), ("Produse în matrice", f"{I['produse']:,}"),
         ("Seturi frecvente", f"{I['seturi_frecvente']:,}"), ("Reguli de asociere", f"{I['reguli']:,}")])
    st.subheader("Profilul coșurilor (o factură = un coș)")
    c1, c2 = st.columns([1, 1])
    with c1:
        arata_tabel("lab5_profil.csv")
    with c2:
        arata_tabel("lab5_top_produse.csv", zecimale=3, titlu="Cele mai frecvente 10 produse")
    arata_fig("lab5_profil")
    observa("Frecvența produselor are coadă lungă (graficul log–log din dreapta): puține produse apar în multe facturi, iar majoritatea apar rar. "
            "De aceea pragul de suport minim decide câte reguli găsim.")
    arata_tabel("curatare.csv", index_col=None, zecimale=2, titlu="Curățarea datelor (comună laboratoarelor)")
    pentru_prezentare(P["taburi"][2])

with t3:
    st.markdown(
        f"**Împărțire în timp:** mining pe **{I['coșuri_train']:,}** coșuri dinainte de {I['data_taiere']}, validare pe **{I['coșuri_test']:,}** coșuri de după. "
        f"Matrice coș × produs cu **{I['produse']}** produse."
    )
    st.subheader("Alegerea suportului minim")
    arata_tabel("lab5_sensibilitate_suport.csv", zecimale=3)
    st.success(f"Suport minim ales: **{I['suport_minim']}** (cel mai mic pentru care rămân cel mult 1.000 de reguli cu confidence ≥ {I['confidence_minim']} și lift > 1).")
    st.subheader("Discretizarea pentru reguli cantitative")
    arata_tabel("lab5_binning.csv", zecimale=0)
    st.caption("Intervalele de lățime egală pun aproape toți clienții în primul interval (coadă lungă); cuantilele îi împart echilibrat.")
    pentru_prezentare(P["taburi"][3])

with t4:
    st.subheader("Cei trei algoritmi pentru seturi frecvente")
    arata_tabel("lab5_algoritmi.csv", zecimale=3)
    st.caption("Aceleași seturi frecvente, timpi diferiți. ECLAT e implementat direct (liste de facturi ca numere pe biți).")

    st.subheader("Reguli de asociere")
    if reguli is not None:
        c1, c2, c3, c4 = st.columns(4)
        ms = c1.slider("support minim", 0.0, float(reguli["support"].max()), 0.0, step=0.001, format="%.3f")
        mc = c2.slider("confidence minim", float(I["confidence_minim"]), 1.0, float(I["confidence_minim"]), step=0.01)
        ml = c3.slider("lift minim", 1.0, float(reguli["lift"].max()), 1.0, step=0.5)
        ordonare = c4.selectbox("Ordonare", ["lift", "confidence", "support"])
        filtrat = reguli[(reguli["support"] >= ms) & (reguli["confidence"] >= mc) & (reguli["lift"] >= ml)].sort_values(ordonare, ascending=False)
        st.caption(f"{len(filtrat)} reguli din {len(reguli)}")
        afiseaza_df(filtrat[["antecedent_text", "consecvent_text", "support", "confidence", "lift"]], inaltime=380)
    arata_fig("lab5_reguli")

    st.subheader("Seturi frecvente (≥ 2 produse)")
    fr = tabel("lab5_seturi_frecvente.csv", index_col=None)
    if fr is not None:
        afiseaza_df(fr[["set", "produse", "suport"]].head(50), inaltime=300)

    st.subheader("Reguli de asociere cantitative (clienți → comportament viitor)")
    q = tabel("lab5_reguli_cantitative.csv", index_col=None)
    if q is not None:
        rez = st.radio("Rezultatul din viitor", sorted(q["consecvent"].unique()), horizontal=True)
        afiseaza_df(q[q["consecvent"] == rez][["antecedent_text", "support", "confidence", "lift", "bază"]].head(30), inaltime=320)
        st.caption("`bază` = ponderea rezultatului în toți clienții; lift = confidence / bază. Regulile descriu clienți, nu sunt un clasificator (vezi Laboratorul 2).")

    st.subheader("Tipare secvențiale {A} → {B}")
    sq = tabel("lab5_secventiale.csv", index_col=None)
    if sq is not None:
        c1, c2 = st.columns(2)
        min_cl = c1.slider("clienți cu ambele produse, minim", 30, int(sq["clienți cu ambele"].max()), 30)
        min_p = c2.slider("„A primul” minim (%)", 50, 100, 60)
        f = sq[(sq["clienți cu ambele"] >= min_cl) & (sq["A primul %"] >= min_p)]
        st.caption(f"{len(f):,} perechi. 50% = fără ordine, doar co-apariție.")
        afiseaza_df(f[["produs A", "produs B", "clienți cu ambele", "A primul %", "support", "confidence"]].head(200), inaltime=320)
    pentru_prezentare(P["taburi"][4])

with t5:
    st.subheader("Rezistă regulile în timp?")
    c1, c2 = st.columns([1, 2])
    with c1:
        arata_tabel("lab5_stabilitate.csv")
    with c2:
        arata_fig("lab5_reguli", "Stânga: support vs confidence (culoare = lift). Dreapta: lift în istoric vs lift în test (pe diagonală = stabil).")
    st.subheader("Utile ca recomandări? (testul „ascunde și ghicește”)")
    arata_tabel("lab5_recomandari.csv", zecimale=1)
    st.success(f"Hit rate@5: reguli (lift) **{I['hit_rate_lift']:.1f}%** față de baza „produse populare” **{I['hit_rate_baza']:.1f}%**. "
               f"{I['lift_test_pastrat']:.0%} dintre reguli păstrează lift > 1 pe date noi.")
    st.caption("Acoperirea mică a regulilor înseamnă că pentru multe coșuri nu există regulă; baza recomandă mereu, dar mai rar nimerește.")
    if reguli is not None:
        st.markdown("**Primele reguli, cu valorile din istoric și din test**")
        afiseaza_df(reguli.head(25)[["antecedent_text", "consecvent_text", "support", "confidence", "lift", "support_test", "confidence_test", "lift_test"]], inaltime=380)
    pentru_prezentare(P["taburi"][5])

with t6:
    st.subheader("Ce recomandăm pentru acest coș?")
    produse = tabel_text("lab5_produse.csv")
    if produse is not None and reguli is not None:
        produse["suport"] = produse["suport"].astype(float)
        produse = produse.sort_values("suport", ascending=False)
        produse["eticheta"] = produse["StockCode"] + " · " + produse["Description"].fillna("")
        eticheta_la_cod = dict(zip(produse["eticheta"], produse["StockCode"]))
        cod_la_desc = dict(zip(produse["StockCode"], produse["Description"].fillna("")))
        c1, c2 = st.columns([3, 1])
        alese = c1.multiselect("Produsele din coș", produse["eticheta"].tolist(), placeholder="Alege unul sau mai multe produse")
        ordine = c2.selectbox("Ordonare reguli", ["lift", "confidence"])
        n = c2.slider("Nr. recomandări", 1, 10, 5)
        if alese:
            cos = {eticheta_la_cod[e] for e in alese}
            r = reguli.sort_values([ordine, "lift"], ascending=False)
            rec, folosite = [], []
            for _, row in r.iterrows():
                ant = set(row["antecedent_cod"].split("|"))
                if ant <= cos and row["consecvent"] not in cos and row["consecvent"] not in rec:
                    rec.append(row["consecvent"]); folosite.append(row)
                    if len(rec) == n:
                        break
            if rec:
                afiseaza_df(pd.DataFrame(folosite)[["antecedent_text", "consecvent_text", "support", "confidence", "lift"]])
            else:
                st.info("Nicio regulă nu se potrivește cu acest coș (la pragul de suport ales). Încearcă produse mai populare.")
    pentru_prezentare(P["taburi"][6])

concluzie(P["concluzie"])
