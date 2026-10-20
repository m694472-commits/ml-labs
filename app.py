import streamlit as st

st.set_page_config(page_title="Machine Learning · Laboratoare", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
[data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
[data-testid="stMetricLabel"] {opacity: .8;}
h1 {font-size: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# notele de prezentare sunt ascunse implicit (publicul nu le vede); le pornești din bara laterală când vrei să le citești
note = st.sidebar.toggle("Note pentru prezentare", value=False, key="note_prezentare", help="Arată blocurile „Pentru prezentare” și pagina cu fișe. Oprește-l când prezinți.")

pagini = st.navigation(
    {
        "Proiect": [
            st.Page("pages/Acasa.py", title="Acasă", icon="🏠", default=True),
            st.Page("pages/0_Datasetul.py", title="Datasetul", icon="🗂️"),
        ],
        "Laboratoare": [
            st.Page("pages/1_Regresia.py", title="1 · Regresia", icon="📈"),
            st.Page("pages/2_Clasificarea.py", title="2 · Clasificarea", icon="🎯"),
            st.Page("pages/3_Clustering.py", title="3 · Clustering", icon="🧩"),
            st.Page("pages/4_Reducerea_dimensionalitatii.py", title="4 · Reducerea dimensionalității", icon="🔬"),
            st.Page("pages/5_Analiza_tiparelor.py", title="5 · Analiza tiparelor", icon="🛒"),
            st.Page("pages/6_Detectarea_anomaliilor.py", title="6 · Detectarea anomaliilor", icon="🚨"),
        ],
        "Încheiere": [st.Page("pages/7_Concluzii.py", title="Concluzii", icon="✅")]
        + ([st.Page("pages/8_Fise_prezentare.py", title="Fișe de prezentare", icon="🎤")] if note else []),
    }
)

with st.sidebar:
    st.divider()
    st.slider("Mărimea imaginilor", 40, 100, 70, step=5, key="scara_img", format="%d%%",
              help="Micșorează sau mărește toate figurile. Pe fiecare imagine, butonul din colț o deschide pe tot ecranul.")

pagini.run()
