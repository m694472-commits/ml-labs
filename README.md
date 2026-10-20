# Machine Learning · Laboratoare (Online Retail II)

Șase laboratoare (regresie, clasificare, clustering, reducerea dimensionalității, analiza tiparelor, detectarea anomaliilor) pe același dataset, în ordinea CRISP-DM, prezentate într-o aplicație Streamlit.

## Rulare locală
```
pip install -r requirements-notebooks.txt
```
1. Rulează notebook-urile în ordine (Restart → Run All): `laborator_0_datasetul`, `laborator_1_2_regresie_clasificare`, `laborator_3_...`, `laborator_4_...`, `laborator_5_...`, `laborator_6_...`. Ele creează folderul `artefacte/` (tabele, figuri, modele).
2. Pornește aplicația: `python -m streamlit run app.py`

Aplicația doar **citește** `artefacte/`; nu recalculează nimic.

## Publicare (GitHub + Streamlit Community Cloud)
1. Pe GitHub: *New repository* (public), apoi încarcă tot conținutul acestui folder, **inclusiv `artefacte/`** (fără el aplicația nu are ce afișa).
2. Pe https://share.streamlit.io: *Create app* → alege depozitul, branch-ul `main`, fișierul principal `app.py`, Python 3.11 → *Deploy*.
3. Linkul rezultat se deschide de pe orice calculator.

Dacă refaci notebook-urile, încarcă din nou folderul `artefacte/` actualizat.
