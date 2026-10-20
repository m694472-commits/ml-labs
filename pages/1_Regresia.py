import numpy as np
import streamlit as st

from prezentare import lab
from utils import (arata_fig, arata_tabel, cere_artefacte, concluzie, construieste_x, formular_client, info, kpi, model, observa,
                   pentru_prezentare, sarcina)

cere_artefacte()
I = info()
R = I["regresie"]
P = lab(1)
zile, taiere = I["zile_fereastra"], I["data_taiere"]

st.title("Laborator 1 · Regresia multifactorială")
st.caption(P["intrebare"])
kpi(P["kpi"])

t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Sarcina", "2 · Datele", "3 · Pregătirea", "4 · Modelarea", "5 · Evaluarea", "6 · Implementarea"])

with t1:
    sarcina(
        intrebare=f"Cât va cheltui în următoarele **{zile} de zile** un client **care cumpără**?",
        unitate=(f"**Unitatea:** clientul (un rând pe client; **{I['clienti_regresie']:,}** clienți activi).\n\n"
                 f"**Ținta (y):** cheltuiala netă = vânzări − retururi în ultimele {zile} de zile.\n\n"
                 f"**Variabilele (x):** ce a făcut clientul *înainte* de {taiere}."),
        criterii=["Bate baza (media) pe R², RMSE, MAE și MAPE.", "Nu supraantrenează: diferență mică între train și test.",
                  "Rezultatul se confirmă prin validare încrucișată."],
        lista_decizii=[
            ("Două probleme în loc de una", "O parte din clienți au ținta 0. Separăm *cumpără?* (Laboratorul 2) de *cât cheltuiește?* (acesta)."),
            (f"Data de tăiere {taiere}", "Variabilele vin din istoric, ținta din viitor: modelul nu vede ce prezice (fără scurgere de informație)."),
            ("Ținta în `log1p`", "Cheltuiala are coadă lungă; câțiva angrosiști ar domina eroarea pătratică."),
            ("Doar clienții activi", "Metrica (R², MAE) are sens pe cheltuieli reale, nu pe o masă de zerouri."),
        ])
    pentru_prezentare(P["taburi"][1])

with t2:
    kpi([("Rânduri brute", f"{I['randuri_brute']:,}"), ("Rânduri de vânzări curate", f"{I['randuri_curate']:,}"),
         ("Clienți cu istoric", f"{I['clienti']:,}"), ("Clienți activi (regresie)", f"{I['clienti_regresie']:,}")])
    arata_tabel("probleme_calitate.csv", index_col=None, titlu="Probleme de calitate găsite în datele brute")
    observa(f"Doar {I['randuri_curate'] / I['randuri_brute']:.1%} din rândurile brute sunt vânzări de produse cu preț și cantitate pozitive. "
            "Documentația UCI spune că `StockCode` are 5 cifre și că doar facturile „C” sunt anulări; în date apar și coduri de servicii "
            "(`POST`, `M`, `BANK CHARGES`…) și facturi „A” (ajustări contabile), care nu sunt vânzări.")
    pentru_prezentare(P["taburi"][2])

with t3:
    st.subheader("Curățare în straturi")
    arata_tabel("curatare.csv", index_col=None, zecimale=2)
    st.caption("Rândurile sunt mutate în straturi, nu șterse: anulările intră în cheltuiala netă, iar serviciile și ajustările sunt studiate în Laboratorul 6.")
    st.subheader("Distribuția țintei")
    arata_fig("tinta_distributie", "Țintă brută (stânga) vs log1p (dreapta): `log1p` face distribuția aproape simetrică.")
    st.subheader("Extreme (IQR și Z-score)")
    arata_tabel("extreme_tinta.csv")
    st.caption("Nu ștergem clienții mari (angrosiști reali, previzibili); reducem influența lor cu `log1p`.")
    st.subheader("Corelații și multicolinearitate")
    arata_fig("corelatie", "Matricea de corelație între variabile și țintă (y_log).")
    arata_tabel("vif.csv", titlu="VIF (peste 10 = variabilă redundantă)")
    observa(f"Variabile scoase ca redundante: `{', '.join(I['redundante_scoase'])}` (identitate: vânzări = frecvență × valoare medie). "
            f"Rămân {len(I['features'])} variabile, toate cu VIF sub 10.", titlu="De ce le scoatem")
    pentru_prezentare(P["taburi"][3])

with t4:
    st.subheader("Zece modele + gradient descent")
    arata_tabel("metrici_regresie_10_modele.csv", csv=True)
    st.caption("R2_train ≫ R2_test la arbori = supraantrenare. `*_cv` = validare încrucișată 5-fold pe train.")
    st.subheader("Coeficienți: liniar vs Ridge vs Lasso")
    arata_tabel("coeficienti_regresie.csv", zecimale=4)
    arata_fig("lasso_traiectorie", "Cu cât λ crește, Lasso aduce coeficienții la 0; cei care rezistă cel mai mult sunt cei mai importanți.")
    pentru_prezentare(P["taburi"][4])

with t5:
    st.subheader("Finaliști și ansambluri")
    arata_tabel("metrici_regresie_finalisti.csv", csv=True)
    st.success(f"Model ales (RMSE minim în validare încrucișată): **{R['model']}** · R² test = {R['r2_test']:.3f} · "
               f"MAE = {R['mae_lire']:.0f} £ · MAPE = {R['mape']:.0f}%")
    arata_fig("regresie_diagnostic", "Real vs prezis, reziduuri, corelația predicțiilor între finaliști.")
    arata_tabel("corelatie_predictii_regresie.csv", titlu="Corelația predicțiilor (aproape de 1 = ansamblul aduce puțin)")
    st.markdown(
        "**Limite:** o singură dată de tăiere; fereastra țintă include sezonul sărbătorilor; clienții fără ID lipsesc; "
        "`expm1` al unei predicții în log estimează mai degrabă mediana, deci erorile în lire sunt ușor părtinitoare."
    )
    pentru_prezentare(P["taburi"][5])

with t6:
    st.subheader("Cât va cheltui acest client?")
    st.caption("Completează istoricul clientului (valorile implicite sunt medianele din date).")
    with st.form("form_reg"):
        v = formular_client("reg")
        trimis = st.form_submit_button("Estimează cheltuiala")
    if trimis:
        X = construieste_x(v, I["features"])
        pred = float(np.expm1(model("model_regresie.joblib").predict(X))[0])
        st.metric("Cheltuiala estimată în următoarele 90 de zile (dacă cumpără)", f"{max(pred, 0):,.0f} £")
        st.caption("Estimare tipică (mediană), nu garantată: cât se poate abate se vede la eroarea medie din tabul 5.")
    pentru_prezentare(P["taburi"][6])

concluzie(P["concluzie"])
