import streamlit as st

from prezentare import NUME_LAB, fisa_md, toate_fisele_md

st.title("Fișe de prezentare")
st.caption("O fișă pe laborator: cifrele-cheie, ce ai făcut, de ce, întrebări posibile, concluzia și un text de vorbit. Toate cifrele vin din ultima rulare.")

st.download_button("⬇ Descarcă toate fișele (.md)", toate_fisele_md().encode("utf-8"), file_name="fise_prezentare.md", mime="text/markdown")

taburi = st.tabs([f"{i} · {NUME_LAB[i]}" for i in range(1, 7)])
for i, t in zip(range(1, 7), taburi):
    with t:
        st.markdown(fisa_md(i))
