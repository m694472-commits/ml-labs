import numpy as np
import streamlit as st

from prezentare import lab
from utils import (arata_fig, arata_tabel, cere_artefacte, concluzie, construieste_x, formular_client, info, kpi, model, observa,
                   pentru_prezentare, sarcina)

cere_artefacte()
I = info()
C = I["clasificare"]
P = lab(2)
zile, taiere = I["zile_fereastra"], I["data_taiere"]

st.title("Laborator 2 · Clasificarea binară")
st.caption(P["intrebare"])
kpi(P["kpi"])

t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Sarcina", "2 · Datele", "3 · Pregătirea", "4 · Modelarea", "5 · Evaluarea", "6 · Implementarea"])

with t1:
    sarcina(
        intrebare=f"Va **cumpăra** clientul cel puțin o dată în următoarele **{zile} de zile**? (da / nu)",
        unitate=(f"**Unitatea:** clientul (**{I['clienti']:,}** clienți cu istoric).\n\n"
                 f"**Eticheta (y):** 1 dacă are cel puțin o vânzare în ultimele {zile} de zile, altfel 0.\n\n"
                 f"**Variabilele (x):** aceleași ca la regresie, doar din istoricul de dinainte de {taiere}."),
        criterii=["ROC-AUC și F1 peste bază (clasa majoritară).", "Confirmat prin validare încrucișată.", "Fără supraantrenare (train ≈ test)."],
        lista_decizii=[
            ("Aceleași variabile ca la regresie", "Cele două laboratoare răspund la întrebări legate (cumpără? și cât?) pe aceleași informații."),
            ("AUC, nu doar acuratețe", "AUC ordonează clienții indiferent de prag; acuratețea depinde de prag și de proporția claselor."),
            ("Pragul se alege separat", "Pragul 0,5 nu e obligatoriu: îl alegem pe validare, în funcție de compromisul precision / recall."),
            ("Utilitate", "Clienții cu probabilitate mică de recumpărare pot primi o campanie de reactivare."),
        ])
    pentru_prezentare(P["taburi"][1])

with t2:
    kpi([("Clienți cu istoric", f"{I['clienti']:,}"), ("Cumpără în 90 de zile", f"{C['pondere_clasa_1']:.1%}"),
         ("Nu cumpără", f"{1 - C['pondere_clasa_1']:.1%}"), ("Rânduri de vânzări curate", f"{I['randuri_curate']:,}")])
    arata_tabel("probleme_calitate.csv", index_col=None, titlu="Probleme de calitate (aceleași date ca la regresie)")
    st.subheader("Balansul claselor")
    arata_tabel("balans_clase.csv")
    observa("Dacă o clasă ar fi foarte rară, acuratețea ar înșela (un model care spune mereu „nu” ar părea bun) și ar trebui SMOTE sau curba precision-recall. "
            "La acest raport al claselor nu rebalansăm artificial, dar urmărim precision, recall, F1 și AUC, nu doar acuratețea.")
    pentru_prezentare(P["taburi"][2])

with t3:
    st.subheader("Curățare în straturi (comună cu regresia)")
    arata_tabel("curatare.csv", index_col=None, zecimale=2)
    st.markdown(f"Tabelul de clienți are **{I['clienti']:,}** rânduri. Împărțire stratificată 80/20, scalare în `Pipeline` (învățată doar pe train).")
    arata_tabel("vif.csv", titlu="VIF: aceleași variabile finale ca la regresie")
    pentru_prezentare(P["taburi"][3])

with t4:
    st.subheader(f"Nouă modele (K la KNN = {C['k_knn']}, ales prin validare încrucișată)")
    arata_tabel("metrici_clasificare_9_modele.csv", csv=True)
    st.caption("Acc_train / ROC_AUC_train mult peste test = supraantrenare (ex. arbori). `*_cv` = validare încrucișată 5-fold pe train.")
    pentru_prezentare(P["taburi"][4])

with t5:
    st.subheader("Finaliști și ansambluri (voting, stacking, bagging)")
    arata_tabel("metrici_clasificare_finalisti.csv", csv=True)
    st.success(f"Model ales (AUC maxim în validare încrucișată): **{C['model']}** · AUC test = {C['auc_test']:.3f} · "
               f"AUC cv = {C['auc_cv']:.3f} · F1 test = {C['f1_test']:.3f}")
    st.subheader("Matricea de confuzie (prag 0,5)")
    arata_fig("clasificare_matrici_confuzie")
    st.subheader("Curbele ROC și Precision-Recall")
    arata_fig("clasificare_roc_pr")
    st.subheader("Pragul de decizie")
    arata_fig("clasificare_prag")
    arata_tabel("prag_comparatie.csv", titlu="Prag implicit vs prag ales (pe test)")
    st.subheader("Ce contează în model")
    arata_fig("clasificare_importanta")
    arata_tabel("corelatie_predictii_clasificare.csv", titlu="Corelația probabilităților prezise între finaliști")
    pentru_prezentare(P["taburi"][5])

with t6:
    st.subheader("Va cumpăra acest client?")
    with st.form("form_clf"):
        v = formular_client("clf")
        trimis = st.form_submit_button("Estimează")
    if trimis:
        X = construieste_x(v, I["features"])
        p = float(model("model_clasificare.joblib").predict_proba(X)[0, 1])
        decizie = p >= C["prag"]
        a, b = st.columns(2)
        a.metric("Probabilitatea de a cumpăra în 90 de zile", f"{p:.1%}")
        b.metric(f"Decizie (prag {C['prag']:.2f})", "Va cumpăra" if decizie else "Nu va cumpăra")
        if decizie:
            cost = float(np.expm1(model("model_regresie.joblib").predict(X))[0])
            st.info(f"Dacă cumpără, cheltuiala estimată (modelul din Laboratorul 1): {max(cost, 0):,.0f} £")
    pentru_prezentare(P["taburi"][6])

concluzie(P["concluzie"])
