import streamlit as st

from prezentare import lab
from utils import cere_artefacte, info, info_optional, kpi

cere_artefacte()
I = info()
I0 = info_optional("info_lab0.json")

st.title("Machine Learning · Laboratoare")
st.caption(f"Dataset: {I['dataset']} · metodologie: CRISP-DM · date până la {I['sfarsit_date']}")

kpi([("Rânduri brute", f"{I['randuri_brute']:,}"), ("Vânzări curate", f"{I['randuri_curate']:,}"),
     ("Clienți cu istoric", f"{I['clienti']:,}"), ("Clienți care cumpără în 90 de zile", f"{I['clasificare']['pondere_clasa_1']:.0%}")])

st.markdown(
    f"""
Datele sunt tranzacțiile unui retailer online britanic. Din ele construim un tabel cu **un rând pe client**: variabilele (recență, frecvență, valoare)
vin din **istoric** (până la {I['data_taiere']}), iar ținta din **următoarele {I['zile_fereastra']} de zile**, ca modelele să nu „vadă” viitorul.
Fiecare laborator urmează fazele **CRISP-DM**: înțelegerea sarcinii → a datelor → pregătirea datelor → modelare → evaluare → implementare.
"""
)

if I0 is not None:
    st.page_link("pages/0_Datasetul.py", label="Despre date: coloane, calitate, curățare, distribuții", icon="🗂️")

st.subheader("Laboratoarele")
pagini = {1: ("pages/1_Regresia.py", "📈"), 2: ("pages/2_Clasificarea.py", "🎯"), 3: ("pages/3_Clustering.py", "🧩"),
          4: ("pages/4_Reducerea_dimensionalitatii.py", "🔬"), 5: ("pages/5_Analiza_tiparelor.py", "🛒"), 6: ("pages/6_Detectarea_anomaliilor.py", "🚨")}
for rand in ((1, 2, 3), (4, 5, 6)):
    cols = st.columns(3)
    for col, nr in zip(cols, rand):
        P = lab(nr)
        with col.container(border=True):
            if P is None:
                st.markdown(f"**Laborator {nr}**")
                st.caption("Notebook-ul nu a fost rulat încă.")
                continue
            st.markdown(f"**Laborator {nr} · {P['titlu'].split(':')[0]}**")
            st.caption(P["intrebare"])
            a = P["kpi"][0]
            st.metric(a[0], a[1])
            st.page_link(pagini[nr][0], label="Deschide laboratorul", icon=pagini[nr][1])

st.caption("În bara din stânga: sliderul „Mărimea imaginilor” (figurile se pot mări și cu butonul de pe imagine), apoi pagina „Concluzii” și „Fișe de prezentare”.")
