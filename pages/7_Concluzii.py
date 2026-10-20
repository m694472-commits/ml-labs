import streamlit as st

from prezentare import NUME_LAB, lab
from utils import info

st.title("Concluzii")
st.caption("Un rezultat-cheie și o concluzie scurtă pentru fiecare laborator.")

rânduri = ["| Laborator | Întrebarea | Rezultatul-cheie | Concluzia |", "|---|---|---|---|"]
for nr in range(1, 7):
    P = lab(nr)
    if P is None:
        rânduri.append(f"| **{nr} · {NUME_LAB[nr]}** | | *notebook nerulat* | |")
        continue
    cheie = " · ".join(f"{e[0]}: **{e[1]}**" for e in P["kpi"][:2])
    rânduri.append(f"| **{nr} · {NUME_LAB[nr]}** | {P['intrebare']} | {cheie} | {P['concluzie']} |")
st.markdown("\n".join(rânduri))

c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("##### Ce leagă laboratoarele")
    I = info()
    st.markdown(
        f"- **Aceleași date, aceeași dată de tăiere** ({I['data_taiere']}): variabilele vin din istoric, ținta din ultimele {I['zile_fereastra']} de zile.\n"
        "- **Testul se folosește o singură dată**; alegerile (model, prag, număr de componente, suport minim) se fac pe validare sau în timp.\n"
        "- **Explicăm, nu doar măsurăm**: fiecare rezultat vine cu o interpretare, iar anomaliile cu motivul pentru care sunt neobișnuite."
    )
with c2.container(border=True):
    st.markdown("##### Limite comune")
    st.markdown(
        "- O singură dată de tăiere și ~2 ani de date (o singură sărbătoare de iarnă pe an).\n"
        "- Clienții fără ID lipsesc din analizele pe client.\n"
        "- Rezultatele descriu asocieri observate în aceste date; nu dovedesc cauze."
    )
