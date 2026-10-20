"""Verifică dacă `artefacte/` conține rezultatele celor mai noi notebook-uri.
Rulare (din ML_labs):  python verifica_artefacte.py
"""
import json
from pathlib import Path

ART = Path(__file__).parent / "artefacte"
ASTEPTATE = {
 "info.json": {
  "notebook": "laborator_1_2_regresie_clasificare",
  "chei": [
   "dataset",
   "randuri_brute",
   "randuri_curate",
   "clienti",
   "clienti_regresie",
   "data_taiere",
   "sfarsit_date",
   "zile_fereastra",
   "features",
   "redundante_scoase",
   "regresie",
   "regresie.model",
   "regresie.rmse_log_cv",
   "regresie.r2_test",
   "regresie.mae_lire",
   "regresie.mape",
   "regresie.r2_cv",
   "regresie.rmse_lire",
   "regresie.r2_train",
   "clasificare",
   "clasificare.model",
   "clasificare.auc_cv",
   "clasificare.auc_test",
   "clasificare.f1_test",
   "clasificare.prag",
   "clasificare.precision",
   "clasificare.recall",
   "clasificare.accuracy",
   "clasificare.k_knn",
   "clasificare.pondere_clasa_1",
   "mediane_clienti"
  ]
 },
 "info_lab0.json": {
  "notebook": "laborator_0_datasetul",
  "chei": [
   "randuri_brute",
   "randuri_curate",
   "facturi_curate",
   "produse",
   "clienti_id",
   "tari",
   "data_min",
   "data_max",
   "data_taiere",
   "zile_fereastra",
   "pct_fara_client",
   "duplicate_exacte",
   "pct_duplicate",
   "pct_retururi",
   "pct_venit_uk",
   "venit_total",
   "clienti_cu_istoric",
   "pct_cumpara_viitor",
   "luna_maxima",
   "pct_venit_luna_maxima",
   "retururi_fara_client"
  ]
 },
 "info_lab3.json": {
  "notebook": "laborator_3_clustering",
  "chei": [
   "model",
   "k",
   "variabile_x",
   "log_pe",
   "brut",
   "brut.R",
   "brut.F",
   "brut.V",
   "silhouette",
   "davies_bouldin",
   "calinski_harabasz",
   "v_measure",
   "nume_segmente",
   "nume_segmente.1",
   "nume_segmente.0",
   "nume_segmente.2",
   "mediane",
   "mediane.recenta_zile",
   "mediane.frecventa",
   "mediane.valoare_medie_factura",
   "k_bun_silhouette",
   "eps_dbscan",
   "min_samples_dbscan",
   "clienti",
   "silhouette_pe_k",
   "silhouette_pe_k.2",
   "silhouette_pe_k.3",
   "silhouette_pe_k.4",
   "silhouette_pe_k.5",
   "silhouette_pe_k.6",
   "silhouette_pe_k.7",
   "silhouette_pe_k.8",
   "silhouette_pe_k.9",
   "silhouette_pe_k.10",
   "ierarhic_cel_mai_mare_cluster",
   "ierarhic_cel_mai_mare_cluster.single",
   "ierarhic_cel_mai_mare_cluster.complete",
   "ierarhic_cel_mai_mare_cluster.average",
   "ierarhic_cel_mai_mare_cluster.centroid",
   "ierarhic_cel_mai_mare_cluster.ward",
   "ierarhic_silhouette",
   "ierarhic_silhouette.single",
   "ierarhic_silhouette.complete",
   "ierarhic_silhouette.average",
   "ierarhic_silhouette.centroid",
   "ierarhic_silhouette.ward",
   "dbscan",
   "dbscan.clustere",
   "dbscan.zgomot_pct",
   "dbscan.silhouette",
   "gmm",
   "gmm.covarianta",
   "gmm.k",
   "gmm.silhouette",
   "gmm.varianta_minima",
   "pct_vanzari_top_segment",
   "pct_clienti_top_segment"
  ]
 },
 "info_lab4.json": {
  "notebook": "laborator_4_reducerea_dimensionalitatii",
  "chei": [
   "dimensiuni",
   "k95",
   "k_cv",
   "auc_cv_kcv",
   "auc_cv_k95",
   "auc_test_kcv",
   "auc_test_toate",
   "auc_test_k95",
   "praguri",
   "praguri.80%",
   "praguri.90%",
   "praguri.95%",
   "praguri.99%",
   "prag_produs",
   "perplexity",
   "clienti",
   "clase",
   "clase.0",
   "clase.1",
   "clase.2",
   "mediana_cheltuiala",
   "lda_acuratete_test",
   "lda_acuratete_train",
   "baza_acuratete",
   "trust_tsne",
   "trust_pca2d",
   "var_pc12"
  ]
 },
 "info_lab5.json": {
  "notebook": "laborator_5_analiza_tiparelor",
  "chei": [
   "coșuri_train",
   "coșuri_test",
   "produse",
   "suport_minim",
   "confidence_minim",
   "max_len",
   "reguli",
   "seturi_frecvente",
   "reguli_cantitative",
   "perechi_secventiale",
   "data_taiere",
   "hit_rate_lift",
   "hit_rate_baza",
   "lift_test_pastrat"
  ]
 },
 "info_lab6.json": {
  "notebook": "laborator_6_detectarea_anomaliilor",
  "chei": [
   "facturi",
   "clienti",
   "contaminare",
   "prag_consens",
   "metode",
   "variabile_factura",
   "mediana_log",
   "mediana_log.log_nr_linii",
   "mediana_log.log_cantitate",
   "mediana_log.log_valoare",
   "mediana_log.log_pret_mediu",
   "mad_log",
   "mad_log.log_nr_linii",
   "mad_log.log_cantitate",
   "mad_log.log_valoare",
   "mad_log.log_pret_mediu",
   "nume_variabile",
   "nume_variabile.log_nr_linii",
   "nume_variabile.log_cantitate",
   "nume_variabile.log_valoare",
   "nume_variabile.log_pret_mediu",
   "anomalii_consens_facturi",
   "anomalii_consens_clienti",
   "stabilitate_if",
   "arima_ordin",
   "zile_anormale",
   "episoade",
   "zile_anormale_fara_calendar",
   "zile_evaluate",
   "caz_gasit",
   "caz_id",
   "caz_metode",
   "caz_de_consens",
   "pret_mediu_pe_factura_mediana",
   "procent_venit_anomalii",
   "limite_iqr_valoare",
   "limite_iqr_valoare.jos",
   "limite_iqr_valoare.sus"
  ]
 }
}
CSV = ["balans_clase.csv", "coeficienti_regresie.csv", "corelatie.csv", "corelatie_predictii_clasificare.csv", "corelatie_predictii_regresie.csv", "curatare.csv", "extreme_tinta.csv", "importanta_variabile_clasificare.csv", "lab0_clienti_variabile.csv", "lab0_coloane.csv", "lab0_esantion_linii.csv", "lab0_forma_clienti.csv", "lab0_lunar.csv", "lab0_probleme.csv", "lab0_separare.csv", "lab0_tari.csv", "lab3_alegere_k.csv", "lab3_asimetrie.csv", "lab3_centroizi_z.csv", "lab3_comparatie.csv", "lab3_corelatie_candidati.csv", "lab3_dbscan.csv", "lab3_extreme.csv", "lab3_gmm_bic.csv", "lab3_ierarhic.csv", "lab3_profil_segmente.csv", "lab3_segmente_clienti.csv", "lab4_blestem.csv", "lab4_clase.csv", "lab4_comparatie.csv", "lab4_cv_k.csv", "lab4_incarcari.csv", "lab4_lda.csv", "lab4_matrice.csv", "lab4_pca_praguri.csv", "lab4_pca_varianta.csv", "lab4_performanta_dimensiuni.csv", "lab4_produse.csv", "lab4_proiectii.csv", "lab4_tsne.csv", "lab4_verificare_pca.csv", "lab5_algoritmi.csv", "lab5_binning.csv", "lab5_produse.csv", "lab5_profil.csv", "lab5_recomandari.csv", "lab5_reguli.csv", "lab5_reguli_cantitative.csv", "lab5_secventiale.csv", "lab5_sensibilitate_suport.csv", "lab5_seturi_frecvente.csv", "lab5_stabilitate.csv", "lab5_top_produse.csv", "lab6_anomalii_clienti.csv", "lab6_anomalii_facturi.csv", "lab6_arima.csv", "lab6_caz_cunoscut.csv", "lab6_cele_mai_mari.csv", "lab6_consens.csv", "lab6_contextuale.csv", "lab6_contextuale_facturi.csv", "lab6_episoade.csv", "lab6_importanta.csv", "lab6_jaccard_clienti.csv", "lab6_jaccard_facturi.csv", "lab6_motive.csv", "lab6_rezumat.csv", "lab6_serie_zilnica.csv", "lab6_tratari.csv", "lab6_univariat.csv", "metrici_clasificare_9_modele.csv", "metrici_clasificare_finalisti.csv", "metrici_regresie_10_modele.csv", "metrici_regresie_finalisti.csv"]


def chei(d, p=""):
    out = []
    for k, v in d.items():
        out.append(p + k)
        if isinstance(v, dict) and k != "mediane_clienti":
            out += chei(v, p + k + ".")
    return out


de_rulat = []
for fisier, e in ASTEPTATE.items():
    cale = ART / fisier
    if not cale.exists():
        print(f"LIPSEȘTE  {fisier}  ->  rulează {e['notebook']}.ipynb")
        de_rulat.append(e["notebook"])
        continue
    lipsa = [c for c in e["chei"] if c not in chei(json.loads(cale.read_text(encoding="utf-8")))]
    if lipsa:
        print(f"VECHI     {fisier}  (lipsesc: {', '.join(lipsa[:6])}{'...' if len(lipsa) > 6 else ''})  ->  rulează {e['notebook']}.ipynb")
        de_rulat.append(e["notebook"])
    else:
        print(f"ok        {fisier}")
lipsa_csv = [c for c in CSV if not (ART / c).exists()]
if lipsa_csv:
    print("\nFișiere CSV lipsă:", ", ".join(lipsa_csv))
print()
if de_rulat or lipsa_csv:
    print("DE RULAT (Restart + Run All, până la ultima celulă):")
    for n in dict.fromkeys(de_rulat):
        print("  -", n + ".ipynb")
else:
    print("Totul este la zi.")
