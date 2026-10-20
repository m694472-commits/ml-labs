"""Textele de prezentare ale aplicației: explicații pe tab, concluzii scurte, fișe și text de vorbit.

Nicio cifră nu este scrisă de mână: toate vin din `artefacte/` (fișierele `info_*.json`), deci textele rămân
corecte după fiecare rulare a notebook-urilor. Afirmațiile calitative descriu metoda, nu un rezultat anume.
"""
from utils import info, info_optional, tabel

NUME_LAB = {1: "Regresia", 2: "Clasificarea", 3: "Clustering", 4: "Reducerea dimensionalității", 5: "Analiza tiparelor", 6: "Detectarea anomaliilor"}


def n(x, z=0):
    """Număr cu separator de mii."""
    return f"{x:,.{z}f}"


def pc(x, z=0):
    """Procent dintr-o proporție (0–1)."""
    return f"{x * 100:.{z}f}%"


class _Tolerant(dict):
    """Cifră lipsă dintr-un `info.json` vechi → «–» în loc de eroare (aplicația nu se oprește)."""
    def __missing__(self, cheie):
        return float("nan")


def _info12():
    """`info.json` dintr-o rulare mai veche poate să nu aibă toate cifrele: le completăm din tabelele finaliștilor."""
    I = dict(info())
    R, C = _Tolerant(I["regresie"]), _Tolerant(I["clasificare"])
    I["regresie"], I["clasificare"] = R, C
    t = tabel("metrici_regresie_finalisti.csv")
    if t is not None and R["model"] in t.index:
        for k, col in [("r2_train", "R2_train"), ("r2_cv", "R2_cv"), ("rmse_lire", "RMSE_£")]:
            if k not in R and col in t.columns:
                R[k] = float(t.loc[R["model"], col])
    t = tabel("metrici_clasificare_finalisti.csv")
    if t is not None and C["model"] in t.index:
        for k, col in [("precision", "Precision"), ("recall", "Recall"), ("accuracy", "Accuracy")]:
            if k not in C and col in t.columns:
                C[k] = float(t.loc[C["model"], col])
    return I


# ──────────────────────────── Laborator 1 ────────────────────────────
def _l1():
    I = _info12()
    R = I["regresie"]
    nf = len(I["features"])
    zile, taiere = I["zile_fereastra"], I["data_taiere"]
    gap = R["r2_train"] - R["r2_test"]
    return {
        "titlu": "Regresia: cât cheltuiește un client",
        "intrebare": f"Cât va cheltui în următoarele {zile} de zile un client care cumpără?",
        "kpi": [("R² pe test", f"{R['r2_test']:.3f}"), ("Eroare medie (MAE)", f"{n(R['mae_lire'])} £"), ("Clienți în model", n(I["clienti_regresie"])),
                ("Variabile folosite", str(nf))],
        "taburi": {
            1: {"spui": f"Prezicem cheltuiala netă a unui client în ultimele {zile} de zile, pornind doar de la ce a făcut înainte de {taiere}. Modelăm doar clienții care cumpără ({n(I['clienti_regresie'])}).",
                "de_ce": ["Problema e împărțită în două: *cumpără?* (Laboratorul 2) și *cât cheltuiește dacă cumpără?* (acesta). O parte din clienți au ținta 0, iar acea masă de zerouri ar domina un singur model.",
                          f"Data de tăiere ({taiere}) separă istoricul (variabile) de viitor (țintă): așa modelul nu „vede” ce trebuie să prezică.",
                          "Succesul înseamnă să bată baza (media) pe R², RMSE, MAE și MAPE, fără supraantrenare, confirmat prin validare încrucișată."],
                "intrebari": [("De ce nu un singur model pentru toți clienții?", "Pentru că sunt două mecanisme diferite: dacă cumperi și cât cumperi. Amestecate, zerourile trag predicția în jos pentru cumpărători.")]},
            2: {"spui": "Datele brute conțin și rânduri care nu sunt vânzări de produse (servicii, anulări, ajustări). Le-am mutat în straturi separate, nu le-am șters.",
                "de_ce": ["Documentația spune că `StockCode` are 5 cifre și că doar facturile „C” sunt anulări; în date apar coduri de servicii și facturi „A”. Am verificat în date, nu am presupus.",
                          "Rândurile mutate rămân disponibile (anulările ajută la cheltuiala netă, ajustările la Laboratorul 6)."],
                "intrebari": [("De ce n-ați șters rândurile problematice?", "Fiecare strat are un rol: anulările scad din cheltuiala netă, iar serviciile și ajustările sunt anomalii de calitate studiate în Laboratorul 6.")]},
            3: {"spui": "Ținta are coadă lungă, așa că o modelăm în `log1p`. Variabilele redundante (identități precum vânzări = frecvență × valoare medie) sunt scoase după VIF.",
                "de_ce": ["`log1p` face distribuția aproape simetrică; fără el, câțiva angrosiști ar domina eroarea pătratică.",
                          "Clienții mari nu sunt șterși: sunt reali și previzibili; le reducem doar influența.",
                          f"VIF peste 10 înseamnă că o variabilă e în mare parte explicată de celelalte (coeficienți instabili): am scos {', '.join('`' + c + '`' for c in I['redundante_scoase'])}, rămân {nf} variabile.",
                          "Scalarea stă în `Pipeline`, deci se învață doar pe train."],
                "intrebari": [("De ce `log1p` și nu ștergeți outlierii?", "Outlierii sunt clienți reali; ștergerea lor ar face modelul fals optimist exact pe clienții care contează. Transformarea păstrează toți clienții."),
                              ("Ce e VIF?", "Cât de mult din varianța unei variabile e explicată de celelalte variabile. Peste 10 = variabila nu aduce informație nouă și destabilizează coeficienții.")]},
            4: {"spui": "Am comparat zece modele (liniare, Ridge, Lasso, ElasticNet, SGD, arbori, ansambluri, KNN), toate în același `Pipeline`, cu validare încrucișată 5-fold pe train.",
                "de_ce": ["Compar mai multe familii ca alegerea să se bazeze pe cifre, nu pe preferință.",
                          "Arborii supraantrenează (R² train ≫ R² test), deci îi reglăm cu `RandomizedSearchCV` și îi comparăm cu versiunile simple.",
                          "Lasso arată ordinea în care renunță la variabile: cele care rezistă cel mai mult sunt cele mai importante."],
                "intrebari": [("De ce validare încrucișată și nu doar test?", "Testul se folosește o singură dată, la final. Alegerea modelului se face pe validare, altfel „împrumut” informație din test.")]},
            5: {"spui": f"Modelul ales este **{R['model']}** (RMSE minim în validare încrucișată). Pe test: R² = {R['r2_test']:.3f}, MAE = {n(R['mae_lire'])} £, MAPE = {R['mape']:.0f}%.",
                "de_ce": ["Alegerea pe validare încrucișată, nu pe test, ca testul să rămână o estimare onestă.",
                          (f"Decalajul R² train − test este {gap:.3f}: " + ("mic, deci fără supraantrenare puternică." if gap < 0.15 else "vizibil, deci există o oarecare supraantrenare.") if gap == gap else "Decalajul R² train − test este în tabelul din fila Evaluare."),
                          "Ansamblurile aduc puțin când predicțiile modelelor sunt foarte corelate (tabelul de corelații)."],
                "intrebari": [("De ce MAPE e mare?", "MAPE împarte eroarea la valoarea reală: clienții cu cheltuială mică produc procente mari chiar la erori mici în lire. De aceea raportăm și MAE în lire și R²."),
                              ("Ce nu poate prezice modelul?", "Factori din afara istoricului: campanii, sezon, schimbări în afacerea clientului. R² arată cât din variație se explică din istoric, nu tot ce se poate întâmpla.")]},
            6: {"spui": "Introduci istoricul unui client și primești cheltuiala estimată pentru următoarele 90 de zile, dacă cumpără.",
                "de_ce": ["Formularul cere variabilele brute și aplică aceleași transformări ca în notebook, deci predicția e consistentă.",
                          "Valoarea e o estimare tipică (mediană), nu o garanție: după `expm1`, un model în log estimează mai degrabă mediana."],
                "intrebari": [("De ce estimarea nu e exactă?", "Pentru că fiecare client e un caz individual; eroarea tipică este MAE-ul din tab 5.")]},
        },
        "concluzie": (f"Din istoric se poate estima cheltuiala viitoare a clienților activi (R² test {R['r2_test']:.2f}; eroare medie {n(R['mae_lire'])} £). "
                      f"Modelul este un instrument de ordonare și planificare, nu o promisiune pentru un client anume."),
        "fisa": {
            "pe_scurt": [f"Țintă: cheltuiala netă în {zile} de zile, doar clienți care cumpără ({n(I['clienti_regresie'])}).",
                         "Curățare în straturi; tabel cu un rând pe client; data de tăiere împotriva scurgerii de informație.",
                         "`log1p` pe țintă și pe variabilele cu coadă lungă; VIF scoate variabilele redundante.",
                         "Zece modele + reglaj, comparate prin validare încrucișată; ansambluri.",
                         f"Ales: {R['model']}; verificat o singură dată pe test."],
            "limite": ["O singură dată de tăiere; fereastra țintă include sezonul sărbătorilor.", "Clienții fără ID nu intră în model."],
        },
        "text_vorbit": (f"În primul laborator am întrebat cât va cheltui un client în următoarele {zile} de zile. Am modelat doar clienții care cumpără, "
                        f"pentru că mulți clienți au cheltuială zero, iar acea masă de zerouri ar fi dominat modelul; a doua jumătate a întrebării, dacă cumpără sau nu, este laboratorul 2. "
                        f"Variabilele vin doar din istoricul de dinainte de {taiere}, ca modelul să nu vadă viitorul. Ținta are coadă lungă, deci am folosit `log1p`; "
                        f"am scos variabilele redundante după VIF și am comparat zece modele prin validare încrucișată. Cel mai bun este {R['model']}, "
                        f"cu R² pe test de {R['r2_test']:.2f} și o eroare medie de {n(R['mae_lire'])} lire. Nu îl prezentăm ca pe o promisiune pentru un client, ci ca pe o estimare tipică."),
    }


# ──────────────────────────── Laborator 2 ────────────────────────────
def _l2():
    I = _info12()
    C = I["clasificare"]
    zile, taiere = I["zile_fereastra"], I["data_taiere"]
    return {
        "titlu": "Clasificarea: va cumpăra clientul?",
        "intrebare": f"Va cumpăra clientul în următoarele {zile} de zile? (da / nu)",
        "kpi": [("ROC-AUC pe test", f"{C['auc_test']:.3f}"), ("F1 pe test", f"{C['f1_test']:.3f}"), ("Prag de decizie", f"{C['prag']:.2f}"),
                ("Clienți care cumpără", pc(C["pondere_clasa_1"]))],
        "taburi": {
            1: {"spui": f"Prezicem dacă un client cumpără cel puțin o dată în următoarele {zile} de zile, pe baza istoricului de dinainte de {taiere}. Ajută la a găsi clienții de reactivat.",
                "de_ce": ["Aceleași variabile și aceeași dată de tăiere ca la regresie, deci cele două laboratoare se pot combina (cumpără? + cât?).",
                          "Succesul: ROC-AUC și F1 peste bază, confirmate prin validare încrucișată."],
                "intrebari": [("De ce AUC și nu doar acuratețea?", "Acuratețea depinde de proporția claselor și de prag; AUC măsoară cât de bine ordonează modelul clienții, indiferent de prag.")]},
            2: {"spui": f"Clasele sunt {pc(C['pondere_clasa_1'])} „cumpără” și {pc(1 - C['pondere_clasa_1'])} „nu cumpără”. Datele brute sunt cele din Laboratorul 1, cu aceleași probleme de calitate.",
                "de_ce": ["Verificăm balansul claselor înainte de orice: dacă ar fi foarte dezechilibrate, acuratețea ar înșela și am folosi SMOTE sau precision-recall.",
                          "Aici clasele sunt suficient de echilibrate, deci nu rebalansăm artificial."],
                "intrebari": [("De ce nu ați folosit SMOTE?", "Nu e nevoie când clasele sunt aproximativ echilibrate; SMOTE ar adăuga puncte sintetice fără câștig.")]},
            3: {"spui": "Împărțire stratificată 80/20 (aceeași proporție de clase în train și test) și scalare în `Pipeline`, învățată doar pe train. Variabilele finale sunt cele de la regresie (după VIF).",
                "de_ce": ["Stratificarea păstrează proporția claselor în ambele seturi, deci metricile sunt comparabile.",
                          "Scalerul învățat doar pe train evită scurgerea statisticilor din test."],
                "intrebari": [("De ce aceleași variabile ca la regresie?", "Ca cele două modele să răspundă la întrebări legate pe aceiași clienți și aceleași informații.")]},
            4: {"spui": f"Am comparat nouă modele (inclusiv KNN cu K = {C['k_knn']}, ales prin validare încrucișată) și am urmărit decalajul train − test.",
                "de_ce": ["Un model cu AUC train mult peste test supraantrenează (de exemplu arborii simpli).",
                          "K la KNN nu are valoare „corectă”: îl alegem comparând AUC în validare încrucișată."],
                "intrebari": [("De ce KNN are K = " + str(C["k_knn"]) + "?", "Pentru că a dat cel mai bun AUC în validare încrucișată pe train; un K foarte mic e sensibil la zgomot.")]},
            5: {"spui": f"Ales: **{C['model']}** (AUC maxim în validare încrucișată). Pe test: AUC = {C['auc_test']:.3f}, F1 = {C['f1_test']:.3f}, precision = {C['precision']:.3f}, recall = {C['recall']:.3f}. Prag de decizie: {C['prag']:.2f}.",
                "de_ce": ["Pragul nu e obligatoriu 0,5: l-am ales pe date de validare, în funcție de ce contează (precision vs recall).",
                          "Matricea de confuzie arată tipurile de greșeli (fals pozitiv = eroare de tip I, fals negativ = eroare de tip II).",
                          "Importanța variabilelor arată ce contribuie la decizie."],
                "intrebari": [("De ce pragul nu este 0,5?", "Pragul controlează compromisul dintre precision și recall. L-am ales pe validare, nu pe test, ca să nu optimizăm pe datele de evaluare."),
                              ("Cât de sigur e modelul?", f"AUC {C['auc_test']:.3f} pe test și {C['auc_cv']:.3f} în validare încrucișată: valorile apropiate arată că rezultatul e stabil.")]},
            6: {"spui": "Introduci istoricul unui client și vezi probabilitatea de cumpărare și decizia; dacă e „va cumpăra”, vezi și cheltuiala estimată din Laboratorul 1.",
                "de_ce": ["Probabilitatea e mai informativă decât decizia: permite alte praguri pentru alte campanii."],
                "intrebari": [("Ce faci cu probabilitatea?", "Clienții cu probabilitate mică pot primi o campanie de reactivare, iar cei cu probabilitate mare sunt tratați diferit.")]},
        },
        "concluzie": (f"Din istoric se poate ordona clar cine e probabil să cumpere (AUC test {C['auc_test']:.2f}). "
                      f"Rămâne o parte neexplicată de factori din afara datelor, deci modelul ghidează campaniile, nu garantează comportamentul."),
        "fisa": {
            "pe_scurt": [f"Etichetă: a cumpărat în {zile} de zile (da / nu); {pc(C['pondere_clasa_1'])} da.",
                         "Aceleași variabile și aceeași dată de tăiere ca la regresie.",
                         "Nouă modele + ansambluri (voting, stacking, bagging); K la KNN prin validare.",
                         f"Ales: {C['model']}, prag {C['prag']:.2f} (ales pe validare).",
                         f"Test o singură dată: AUC {C['auc_test']:.3f}, F1 {C['f1_test']:.3f}."],
            "limite": ["O singură dată de tăiere.", "Probabilitățile nu sunt calibrate explicit."],
        },
        "text_vorbit": (f"În al doilea laborator am prezis dacă un client cumpără în următoarele {zile} de zile. Am folosit aceleași variabile și aceeași dată de tăiere ca la regresie, "
                        f"ca cele două laboratoare să se completeze. Clasele sunt suficient de echilibrate ({pc(C['pondere_clasa_1'])} cumpără), deci nu am rebalansat. "
                        f"Am comparat nouă modele și ansambluri prin validare încrucișată; câștigă {C['model']}, cu AUC de {C['auc_test']:.2f} pe test, valoare apropiată de cea din validare. "
                        f"Pragul de decizie, {C['prag']:.2f}, l-am ales pe validare, nu pe test. Rezultatul ordonează bine clienții, dar nu garantează comportamentul unui client anume."),
    }


# ──────────────────────────── Laborator 3 ────────────────────────────
def _l3():
    I = info_optional("info_lab3.json")
    if not I:
        return None
    prof = tabel("lab3_profil_segmente.csv")
    rate = (f"ratele de cumpărare din viitor variază între {prof['% a cumpărat în viitor'].min():.0f}% și {prof['% a cumpărat în viitor'].max():.0f}% de la un segment la altul"
            if prof is not None else "segmentele diferă în ratele de cumpărare din viitor")
    return {
        "titlu": "Clustering: segmente de clienți",
        "intrebare": "Ce grupuri de clienți există, după cum cumpără?",
        "kpi": [("Algoritm ales", I["model"]), ("Silhouette", f"{I['silhouette']:.3f}"), ("Davies-Bouldin", f"{I['davies_bouldin']:.3f}"),
                ("Segment de top", f"{I['pct_clienti_top_segment']:.0f}% din clienți → {I['pct_vanzari_top_segment']:.0f}% din vânzări")],
        "taburi": {
            1: {"spui": "Căutăm grupuri de clienți cu comportament asemănător, fără nicio țintă. Judecăm rezultatul pe trei criterii: separare internă, interpretabilitate și diferențe în comportamentul viitor.",
                "de_ce": ["Fără țintă nu avem acuratețe; de aceea cerem și ca segmentele să se poată descrie în cuvinte și să difere în viitor (date nevăzute de algoritm).",
                          "Aplicația practică: marketing direcționat."],
                "intrebari": [("Cum știi că segmentele sunt bune?", "Intern: silhouette, Davies-Bouldin, Calinski-Harabasz. Extern: segmentele diferă în comportamentul din ultimele 90 de zile, pe care algoritmul nu l-a văzut.")]},
            2: {"spui": "Variabilele sunt R (recență), F (frecvență), V (valoarea medie a unei facturi). M (vânzări totale) este exact F × V, deci nu aduce informație nouă.",
                "de_ce": ["Variabilele cu asimetrie > 1 primesc `log1p`, pentru că distanța euclidiană ar fi dominată de coada lungă.",
                          "Corelația dintre candidate confirmă că M repetă F și V."],
                "intrebari": [("De ce RFM?", "Sunt trei variabile ușor de explicat unui om de marketing: cât de recent, cât de des și cât valorează în medie.")]},
            3: {"spui": f"{n(I['clienti'])} clienți, variabile standardizate (`StandardScaler`). Extremele nu se șterg.",
                "de_ce": ["Standardizarea e obligatorie la metode bazate pe distanță: altfel variabila cu scara mare ar decide singură.",
                          "Clienții cu valori mari sunt exact clienții valoroși pe care segmentarea trebuie să-i găsească."],
                "intrebari": [("De ce nu ștergeți extremele?", "Pentru clustering ele sunt informație: segmentul „valoare mare” există tocmai datorită lor.")]},
            4: {"spui": f"K-means cu K ales după silhouette maxim în intervalul 3–8 (K = {I['k_bun_silhouette']}), apoi ierarhic, DBSCAN și GMM ca verificare.",
                "de_ce": ["Cotul și silhouette se folosesc împreună: cotul e vizual și ambiguu, silhouette dă o valoare.",
                          f"Ierarhic: legăturile single/average dau un cluster uriaș (cel mai mare ia {I['ierarhic_cel_mai_mare_cluster']['single']:.0f}% din clienți), adică efectul de lanț: silhouette mare, dar segmente inutile.",
                          f"DBSCAN: ε din cotul graficului k-distanță ({I['eps_dbscan']:.2f}) dă {I['dbscan']['clustere']} clustere și {I['dbscan']['zgomot_pct']:.1f}% zgomot; în date nu apar regiuni clar separate de densitate mică.",
                          "GMM: alegem forma covarianței după BIC; am fixat o varianță minimă pentru a evita componente degenerate pe variabile discrete (frecvența)."],
                "intrebari": [("De ce nu alegeți ierarhicul cu silhouette mare?", "Silhouette mare la single/average vine din efectul de lanț: aproape toți clienții într-un cluster. Un segment care conține aproape toată baza nu ajută marketingul."),
                              ("De ce DBSCAN nu e ales?", "Datele formează un nor continuu, fără regiuni separate de densitate mică; DBSCAN fie lipește tot, fie produce multe clustere mici.")]},
            5: {"spui": f"Ales: **{I['model']}**, silhouette {I['silhouette']:.3f}, Davies-Bouldin {I['davies_bouldin']:.3f}, Calinski-Harabasz {n(I['calinski_harabasz'])}. Segmentele diferă mult în comportamentul viitor.",
                "de_ce": ["Dintre algoritmii care pot segmenta clienți noi (K-means, GMM) alegem silhouette maxim.",
                          "Numele segmentelor vin din reguli pe terțile variabilelor, nu sunt alese manual.",
                          f"Segmentul de top ({I['pct_clienti_top_segment']:.0f}% din clienți) aduce {I['pct_vanzari_top_segment']:.0f}% din vânzări."],
                "intrebari": [("De ce V-measure e mic?", "V-measure compară segmentele cu o singură etichetă binară (a cumpărat sau nu). Segmentele nu au fost făcute pentru asta; contează că diferă în comportamentul viitor: " + rate + " (tabelul de profil)."),
                              ("Segmentele sunt „reale”?", "Datele formează mai degrabă un continuum: segmentele sunt o convenție utilă de marketing, nu clustere naturale separate.")]},
            6: {"spui": "Introduci recența, frecvența și valoarea medie a unui client și vezi segmentul și profilul lui.",
                "de_ce": ["Folosim același model și aceleași transformări (`log1p`, standardizare) ca în notebook."],
                "intrebari": [("Poate fi folosit pentru clienți noi?", "Da, pentru K-means și GMM; DBSCAN și ierarhicul nu au `predict`.")]},
        },
        "concluzie": (f"Clienții se pot împărți în {I['k']} segmente interpretabile, iar segmentul de top ({I['pct_clienti_top_segment']:.0f}% din clienți) aduce {I['pct_vanzari_top_segment']:.0f}% din vânzări. "
                      "Datele formează un nor continuu, deci segmentele sunt o convenție utilă pentru marketing, nu grupuri naturale separate."),
        "fisa": {
            "pe_scurt": ["Variabile RFM (recență, frecvență, valoare medie a facturii); M exclus (= F × V).", "`log1p` unde asimetria > 1, apoi standardizare.",
                         f"K-means, K = {I['k_bun_silhouette']} (silhouette maxim în 3–8); ierarhic, DBSCAN și GMM ca verificare.",
                         "Segmente denumite prin reguli pe terțile variabilelor.", "Evaluare externă: ce fac segmentele în ultimele 90 de zile."],
            "limite": ["Date fără grupuri separate clar: segmentele sunt convenții.", "Rezultatul depinde de variabilele alese (RFM)."],
        },
        "text_vorbit": (f"În laboratorul 3 am căutat segmente de clienți fără nicio țintă. Am folosit recența, frecvența și valoarea medie a facturii; vânzările totale le-am exclus, "
                        f"pentru că sunt exact produsul dintre frecvență și valoare. Am aplicat log unde distribuția era înclinată și am standardizat. "
                        f"K-means cu K = {I['k']} a avut cel mai bun silhouette, {I['silhouette']:.2f}. Ierarhicul cu legături single sau average pare mai bun la silhouette, dar pune aproape toți clienții "
                        f"într-un cluster, deci nu ajută marketingul. Segmentul de top reprezintă {I['pct_clienti_top_segment']:.0f}% din clienți și {I['pct_vanzari_top_segment']:.0f}% din vânzări. "
                        "Spun deschis că datele formează un nor continuu: segmentele sunt o convenție utilă, nu grupuri naturale."),
    }


# ──────────────────────────── Laborator 4 ────────────────────────────
def _l4():
    I = info_optional("info_lab4.json")
    if not I:
        return None
    D, k95, kcv = I["dimensiuni"], I["k95"], I["k_cv"]
    mai_putine = kcv < k95
    return {
        "titlu": "Reducerea dimensionalității",
        "intrebare": "Putem rezuma ce cumpără un client (sute de produse) în câteva dimensiuni?",
        "kpi": [("Dimensiuni inițiale", n(D)), ("PCA, regula 95%", f"{k95} comp."), ("PCA, ales prin validare", f"{kcv} comp."),
                ("t-SNE trustworthiness", f"{I['trust_tsne']:.3f}", f"PCA 2D: {I['trust_pca2d']:.3f}")],
        "taburi": {
            1: {"spui": f"Matrice client × produs cu {n(I['clienti'])} clienți și {n(D)} produse (dimensiuni). Vrem un rezumat în puține dimensiuni care păstrează informația utilă.",
                "de_ce": ["Motive: eficiență, mai puțină supraantrenare, vizualizare în 2D, mai puțin zgomot.",
                          "Verificăm pe date **blestemul dimensionalității**, nu îl luăm de bun."],
                "intrebari": [("De ce reducem dimensiunile?", "Cu multe dimensiuni distanțele devin asemănătoare și modelele supraantrenează; cu puține, calculul e rapid și putem vizualiza.")]},
            2: {"spui": f"Păstrăm produsele cumpărate de cel puțin {pc(I['prag_produs'])} dintre clienți; tabelul arată ce parte din istoric rămâne acoperită. Clasele de comportament viitor (0 = nu cumpără, 1 = cheltuială ≤ mediană, 2 = peste) sunt pentru LDA și evaluare.",
                "de_ce": ["Produsele cumpărate de foarte puțini clienți sunt coloane aproape goale: adaugă zgomot, nu informație.",
                          f"Mediana ({n(I['mediana_cheltuiala'])} £) e calculată doar pe cumpărători, ca clasele 1 și 2 să fie egale."],
                "intrebari": [("De ce pragul de 2%?", "Compromis: păstrează produsele cu suficienți cumpărători și reduce mult dimensiunea; acoperirea istoricului e raportată ca decizia să aibă cifre.")]},
            3: {"spui": "Împărțire 80/20 stratificată înainte de scalare și PCA, apoi standardizare. Măsurăm cum se uniformizează distanțele când adăugăm dimensiuni.",
                "de_ce": ["Scalerul și componentele se învață pe train, altfel testul „vede” datele.",
                          "PCA e sensibil la scară: fără standardizare, produsele vândute în cantități mari ar domina.",
                          "Coeficientul de variație al distanțelor scade cu dimensiunile: „cel mai apropiat vecin” spune tot mai puțin."],
                "intrebari": [("Ce e blestemul dimensionalității?", "Când dimensiunile cresc, distanțele dintre puncte devin foarte asemănătoare; metodele bazate pe distanță își pierd sensul.")]},
            4: {"spui": f"PCA făcut de mână (covarianță → vectori proprii) și verificat cu sklearn. Regula de 95% cere {k95} componente; validarea încrucișată pe train alege {kcv}.",
                "de_ce": ["95% din varianță e o convenție care nu ține cont de scopul modelului; o comparăm cu o alegere bazată pe performanță.",
                          "PCA se refit în interiorul fiecărui fold, deci validarea nu are scurgere.",
                          "LDA e supervizat și poate învăța „pe de rost” în multe dimensiuni: îl verificăm pe test și după PCA.",
                          "t-SNE e doar pentru vizualizare; îl rulăm pe primele 50 de componente PCA (mai rapid, mai puțin zgomot) și comparăm perplexitățile prin trustworthiness."],
                "intrebari": [("De ce k din validare și nu 95%?", (f"Validarea alege {kcv} componente, mai puține decât {k95}, cu AUC pe test {I['auc_test_kcv']:.3f} față de {I['auc_test_k95']:.3f}." if mai_putine
                                                                  else f"Validarea alege {kcv} componente; regula de 95% cere {k95}.") + " Dincolo de un punct, componentele suplimentare adaugă zgomot, iar modelul supraantrenează."),
                              ("De ce nu comparați KL între perplexități?", "KL nu e comparabil între perplexități diferite; folosim trustworthiness, care măsoară cât de bine rămân vecinii.")]},
            5: {"spui": (f"PCA cu {kcv} componente dă AUC pe test {I['auc_test_kcv']:.3f}, față de {I['auc_test_toate']:.3f} cu toate cele {n(D)} variabile. " if I['auc_test_kcv'] > I['auc_test_toate'] else
                         f"PCA cu {kcv} componente dă AUC pe test {I['auc_test_kcv']:.3f}, față de {I['auc_test_toate']:.3f} cu toate variabilele. ") +
                        f"LDA: acuratețe test {I['lda_acuratete_test']:.3f} (bază {I['baza_acuratete']:.3f}). t-SNE: trustworthiness {I['trust_tsne']:.3f} față de {I['trust_pca2d']:.3f} la PCA 2D.",
                "de_ce": ["Judecăm LDA pe test: pe train acuratețea e mai mare, ceea ce arată supraantrenare, nu separare.",
                          "În graficele 2D, culorile n-au fost folosite de PCA și t-SNE (nesupervizate).",
                          "În t-SNE, distanțele dintre grupuri și mărimea lor nu se interpretează."],
                "intrebari": [("t-SNE e mai bun decât PCA?", "Păstrează mai bine vecinătățile locale (trustworthiness mai mare), dar nu poate proiecta clienți noi și distanțele globale nu au sens. PCA e cel folosit în producție.")]},
            6: {"spui": "Alegi produsele cumpărate și vezi unde cade clientul în harta PCA, față de ceilalți clienți.",
                "de_ce": ["Proiecția folosește pipeline-ul salvat (scalare + PCA), identic cu cel din notebook.",
                          "Nu afișăm o clasă prezisă: scopul reducerii e rezumarea, iar clasificarea vine din Laboratorul 2."],
                "intrebari": [("Se poate folosi t-SNE pentru un client nou?", "Nu: t-SNE nu are `transform`; ar trebui refăcut pe toate datele.")]},
        },
        "concluzie": (f"Din {n(D)} produse, {kcv} componente PCA păstrează informația utilă pentru un model, mai puține decât cele {k95} cerute de regula de 95%. "
                      "Reducerea ajută la vizualizare și la stabilitate; t-SNE servește doar explorării."),
        "fisa": {
            "pe_scurt": [f"Matrice client × produs: {n(I['clienti'])} × {n(D)}.", "Split 80/20 stratificat înainte de scalare și PCA; măsurăm blestemul dimensionalității.",
                         f"PCA de mână vs sklearn; k: regula 95% = {k95}, validare = {kcv}.", "LDA pe toate variabilele vs după PCA, judecat pe test.", "t-SNE pe 50 de componente; trustworthiness pentru perplexity."],
            "limite": ["Produsele rare sunt excluse.", "Clasele depind de o singură fereastră de 90 de zile."],
        },
        "text_vorbit": (f"În laboratorul 4 am pornit de la o matrice client × produs cu {n(D)} dimensiuni. Am verificat mai întâi blestemul dimensionalității: distanțele dintre clienți se uniformizează odată cu dimensiunile. "
                        f"PCA cu regula clasică de 95% din varianță cere {k95} componente, dar când am ales numărul de componente prin validare încrucișată, pe train, au ieșit {kcv}. "
                        f"LDA l-am judecat pe test, nu pe train, pentru că în multe dimensiuni învață pe de rost. t-SNE e doar pentru vizualizare: trustworthiness {I['trust_tsne']:.2f} față de {I['trust_pca2d']:.2f} la PCA 2D, "
                        "dar nu poate proiecta clienți noi, așa că în aplicație proiectăm clienții noi cu PCA."),
    }


# ──────────────────────────── Laborator 5 ────────────────────────────
def _l5():
    I = info_optional("info_lab5.json")
    if not I:
        return None
    return {
        "titlu": "Analiza tiparelor: ce se cumpără împreună",
        "intrebare": "Ce produse se cumpără împreună și ce recomandări putem face?",
        "kpi": [("Reguli de asociere", n(I["reguli"])), ("Suport minim", str(I["suport_minim"])), ("Hit rate@5 (reguli)", f"{I['hit_rate_lift']:.1f}%"),
                ("Hit rate@5 (baza „populare”)", f"{I['hit_rate_baza']:.1f}%")],
        "taburi": {
            1: {"spui": "Căutăm produse care apar împreună în aceeași factură și reguli „dacă A, atunci B” pentru recomandări, plasarea produselor și pachete.",
                "de_ce": ["Support = cât de des apare; confidence = P(B | A); lift = confidence / support(B) (1 = independență, > 1 = asociere pozitivă).",
                          "Succes: cei trei algoritmi dau aceleași seturi, regulile rezistă în timp și bat recomandarea „produse populare”."],
                "intrebari": [("De ce lift și nu doar confidence?", "Un produs foarte popular are confidence mare cu orice. Lift împarte la popularitatea lui B și arată dacă asocierea e reală.")]},
            2: {"spui": "O factură = un coș. Profilul arată câte produse are un coș, cât de lungă e coada de produse rare și cât de mult dominează cele mai frecvente produse.",
                "de_ce": ["Frecvența produselor are coadă lungă (scara log–log): puține produse foarte frecvente, multe rare.",
                          "Asta explică de ce suportul minim trebuie ales atent."],
                "intrebari": [("De ce factura ca unitate?", "Un coș cumpărat odată reflectă produsele care se aleg împreună; clientul pe tot istoricul ar amesteca perioade diferite.")]},
            3: {"spui": f"Împărțire în timp: tipare căutate în {n(I['coșuri_train'])} coșuri dinainte de {I['data_taiere']}, validate pe {n(I['coșuri_test'])} coșuri de după. Matrice coș × produs cu {I['produse']} produse.",
                "de_ce": [f"Suport minim ales: {I['suport_minim']} (cel mai mic pentru care rămân cel mult 1.000 de reguli cu confidence ≥ {I['confidence_minim']} și lift > 1). Un număr pe care un analist îl poate verifica.",
                          "Pragurile sunt punctul sensibil: prea mici dau milioane de tipare, prea mari pierd produsele relevante.",
                          "Pentru regulile cantitative, intervalele pe cuantile împart clienții echilibrat, iar cele de lățime egală pun aproape toți clienții într-un singur interval."],
                "intrebari": [("De ce validare în timp?", "O regulă găsită și testată pe aceleași date arată doar cât de bine le memorează. Testul pe perioada următoare arată dacă regula rămâne valabilă.")]},
            4: {"spui": "Apriori, FP-Growth și ECLAT (implementat direct) dau aceleași seturi frecvente, la timpi diferiți. Din ele generăm reguli, reguli cantitative pe clienți și tipare secvențiale de lungime 2.",
                "de_ce": ["Cele trei algoritmi trebuie să dea exact aceleași seturi: asta verifică corectitudinea implementării.",
                          "Regulile au un singur produs în consecvent, forma folosită la recomandări; păstrăm doar lift > 1 și confidence ≥ 0,5.",
                          "Regulile între variante ale aceluiași produs (culori dintr-o serie) au lift mare, dar sunt evidente și nu ajută recomandările."],
                "intrebari": [("Care e diferența dintre algoritmi?", "Apriori generează candidați nivel cu nivel; FP-Growth comprimă coșurile într-un arbore și evită candidații; ECLAT lucrează cu liste de facturi pe produs. Rezultatul e același, timpul diferă.")]},
            5: {"spui": f"Regulile rezistă în timp ({pc(I['lift_test_pastrat'])} păstrează lift > 1 pe date noi) și ca recomandări ating hit rate@5 de {I['hit_rate_lift']:.1f}%, față de {I['hit_rate_baza']:.1f}% pentru produsele populare.",
                "de_ce": ["Testul „ascunde un produs din coș și ghicește-l” măsoară direct utilitatea ca recomandare.",
                          "Acoperirea regulilor e parțială: pentru multe coșuri nu există nicio regulă, în timp ce baza recomandă mereu, dar rar nimerește."],
                "intrebari": [("Ce înseamnă hit rate@5?", "În câte coșuri, cu un produs ascuns, produsul ascuns apare printre primele 5 recomandări."),
                              ("De ce acoperirea e mică?", "O regulă se aplică doar când toate produsele din antecedent sunt în coș; praguri de suport mari lasă puține reguli.")]},
            6: {"spui": "Alegi produse în coș și primești recomandări pe baza regulilor, cu support, confidence și lift afișate.",
                "de_ce": ["Recomandările vin direct din reguli, deci fiecare se poate explica.", "Dacă nicio regulă nu se potrivește, spunem asta în loc să inventăm o recomandare."],
                "intrebari": [("De ce unele coșuri nu primesc recomandări?", "Pentru că nicio regulă nu are antecedentul inclus în acel coș, la pragul ales.")]},
        },
        "concluzie": (f"Există tipare de co-cumpărare stabile în timp, iar recomandările din reguli ating hit rate@5 de {I['hit_rate_lift']:.1f}%, față de {I['hit_rate_baza']:.1f}% la baza „populare”. "
                      "Acoperirea este parțială: regulile funcționează acolo unde există, iar altfel nu recomandăm nimic."),
        "fisa": {
            "pe_scurt": ["Unitatea: factura (coș). Mining pe istoric, validare pe perioada următoare.",
                         f"Suport minim {I['suport_minim']}, ales să rămână ≤ 1.000 de reguli.", "Apriori = FP-Growth = ECLAT (aceleași seturi).",
                         "Reguli A ⇒ c, reguli cantitative pe clienți, secvențe de lungime 2.", "Utilitate: hit rate@5 față de baza „produse populare”."],
            "limite": ["Reguli evidente între variante ale aceluiași produs.", "Acoperire parțială; recomandările nu țin cont de preț sau stoc."],
        },
        "text_vorbit": (f"În laboratorul 5 am căutat produse care se cumpără împreună. Unitatea este factura. Am căutat tiparele pe coșurile dinainte de {I['data_taiere']} și le-am verificat pe cele de după. "
                        f"Suportul minim, {I['suport_minim']}, l-am ales ca să rămână cel mult o mie de reguli, un număr verificabil de un analist. Apriori, FP-Growth și ECLAT dau aceleași seturi frecvente, doar timpii diferă. "
                        f"Ca recomandări, regulile nimeresc produsul ascuns în {I['hit_rate_lift']:.1f}% din cazuri, față de {I['hit_rate_baza']:.1f}% pentru produsele populare. "
                        "Limita este acoperirea: nu pentru orice coș există o regulă."),
    }


# ──────────────────────────── Laborator 6 ────────────────────────────
def _l6():
    I = info_optional("info_lab6.json")
    if not I:
        return None
    cm = I["caz_metode"]
    caz = (f"Comanda de 80.995 de bucăți (factura {I['caz_id']}) este marcată de {len(cm)} din {len(I['metode'])} metode"
           + (" și intră în consens." if I["caz_de_consens"] else " (sub pragul de consens).")) if I["caz_gasit"] else "Cazul cunoscut nu există în acest set de date."
    return {
        "titlu": "Detectarea anomaliilor",
        "intrebare": "Ce facturi, clienți și zile se abat atât de mult de la normal încât merită verificate?",
        "kpi": [("Facturi de consens", n(I["anomalii_consens_facturi"])), ("Clienți de consens", n(I["anomalii_consens_clienti"])),
                ("Zile anormale", f"{I['zile_anormale']} din {I['zile_evaluate']}"), ("Stabilitate Isolation Forest", f"{I['stabilitate_if']:.3f}")],
        "taburi": {
            1: {"spui": f"Căutăm facturi, clienți și zile neobișnuite, fără etichete. Bugetul de investigare este {pc(I['contaminare'])} din observații.",
                "de_ce": ["O anomalie se investighează, nu se șterge automat: poate fi o eroare sau un client angrosist real.",
                          "Fără etichete nu putem calcula acuratețea; succesul înseamnă acord între metode independente, explicații, găsirea unui caz cunoscut și valoarea adăugată de context."],
                "intrebari": [("Cum validezi fără etichete?", "Prin acordul între șase metode independente, stabilitate la altă inițializare, explicații pentru fiecare anomalie și un caz cunoscut.")]},
            2: {"spui": "Metode grafice și statistice pe valoarea facturii: IQR și Z-score, brut și după `log1p`.",
                "de_ce": ["Z-score pe date asimetrice se „maschează”: media și abaterea standard sunt trase de extreme, deci detectează mai puține.",
                          "IQR e mai robust, iar `log1p` schimbă mult rezultatul."],
                "intrebari": [("De ce nu doar Z-score?", "Pe o distribuție cu coadă lungă, extremele măresc abaterea standard și se ascund singure.")]},
            3: {"spui": "Nu tratăm extremele (nu le ștergem, nu le limităm), pentru că ele sunt exact ținta detecției. Folosim doar `log1p` și standardizare.",
                "de_ce": ["Tabelul compară ștergerea, trunchierea, winsorizarea, `log1p` și Z-score ca să se vadă efectul asupra mediei, abaterii și asimetriei.",
                          "Tratarea extremelor e pasul potrivit înainte de un model predictiv (Laboratorul 1), nu înainte de detecție."],
                "intrebari": [("De ce tratați extremele la regresie și nu aici?", "La regresie vrem un model stabil; aici extremele sunt obiectul căutării.")]},
            4: {"spui": f"Șase metode cu același buget ({pc(I['contaminare'])}): Isolation Forest, LOF, DBSCAN, One-Class SVM, MCD, kNN. Plus context (țara, clientul însuși) și o serie de timp pe venitul zilnic.",
                "de_ce": ["Metodele văd lucruri diferite (izolare, densitate locală, distanță robustă), de aceea cerem consens.",
                          "Contextul arată anomalii pe care analiza globală nu le vede: o factură normală global poate fi neobișnuită pentru clientul ei.",
                          f"Seria de timp: ARIMA sezonier cu variabile de calendar (sărbători UK). Fără calendar sunt {I['zile_anormale_fara_calendar']} zile anormale, cu calendar {I['zile_anormale']}: o parte din „anomalii” sunt abateri previzibile.",
                          "Reziduurile se standardizează robust (mediană și MAD)."],
                "intrebari": [("De ce variabile de calendar?", "Sărbătorile și perioadele libere sunt abateri previzibile, cunoscute dinainte. Fără ele modelul le-ar marca drept anomalii, adică exact ce nu vrem să investigăm."),
                              ("De ce 1%?", "Este o alegere de business (cât putem verifica manual), nu o proprietate a datelor.")]},
            5: {"spui": f"Consens = marcat de cel puțin {I['prag_consens']} din {len(I['metode'])} metode. Rezultă {n(I['anomalii_consens_facturi'])} facturi și {n(I['anomalii_consens_clienti'])} clienți, cu explicație pentru fiecare. Isolation Forest e stabil la 5 inițializări (Jaccard {I['stabilitate_if']:.3f}). {caz}",
                "de_ce": ["Jaccard mic între două metode înseamnă că ele văd alte tipuri de anomalii; de aceea nu alegem una singură.",
                          f"Anomaliile de consens reprezintă {I['procent_venit_anomalii']:.1f}% din venit: contează pentru business.",
                          "Explicația vine din Z robust pe fiecare variabilă: spune ce o face neobișnuită și în ce sens."],
                "intrebari": [("Cum știi că nu sunt doar erori?", "Nu știm: de aceea le marcăm și le explicăm. Unele sunt greșeli, altele clienți angrosiști reali.")]},
            6: {"spui": "Introduci numărul de linii, cantitatea și valoarea unei facturi noi și primești verdictul și explicația, pe variabile.",
                "de_ce": ["Modelul este Isolation Forest cu scalare, antrenat pe toate facturile; prețul mediu se calculează din valoare și cantitate.",
                          "Explicația folosește același Z robust ca în notebook."],
                "intrebari": [("Verdictul e o certitudine?", "Nu: scorul spune cât de izolată e factura. Rezultatul este un motiv de verificare, nu o sentință.")]},
        },
        "concluzie": (f"Cu șase metode și un buget de {pc(I['contaminare'])}, rezultă {n(I['anomalii_consens_facturi'])} facturi și {n(I['anomalii_consens_clienti'])} clienți de consens, fiecare cu explicație. "
                      "Anomaliile nu se șterg, se investighează: unele sunt erori, altele sunt clienți angrosiști reali."),
        "fisa": {
            "pe_scurt": [f"Șase metode punctuale, același buget ({pc(I['contaminare'])}); consens ≥ {I['prag_consens']}.", "Context: față de țară și față de clientul însuși.",
                         "Colective: serie zilnică cu ARIMA sezonier și calendar UK.", "Validare fără etichete: acord (Jaccard), stabilitate, explicații, caz cunoscut.", caz],
            "limite": ["Bugetul de 1% este o alegere de business.", "Seria de timp are ~2 ani: nu captează sezonalitatea anuală completă."],
        },
        "text_vorbit": (f"În laboratorul 6 am căutat facturi, clienți și zile neobișnuite. Nu avem etichete, deci nu calculăm acuratețea; ne bazăm pe acordul dintre șase metode, pe stabilitate și pe explicații. "
                        f"Anomalia nu se șterge, se investighează, pentru că poate fi o eroare sau un client angrosist real. Consensul, cel puțin {I['prag_consens']} din {len(I['metode'])} metode, dă {n(I['anomalii_consens_facturi'])} facturi. "
                        f"{caz} Pentru zilele neobișnuite am folosit o serie de timp cu calendarul sărbătorilor: fără calendar, modelul ar fi marcat ca anomalii chiar zilele libere. "
                        "Rezultatul este o listă de verificat, nu o sentință."),
    }


_CONSTRUCTORI = {1: _l1, 2: _l2, 3: _l3, 4: _l4, 5: _l5, 6: _l6}


def lab(nr):
    """Conținutul de prezentare pentru laboratorul `nr` (1–6); None dacă notebook-ul nu a fost rulat."""
    try:
        return _CONSTRUCTORI[nr]()
    except FileNotFoundError:
        return None


def bloc(nr, tab):
    """Blocul „Pentru prezentare” al unui tab (1–6) dintr-un laborator."""
    c = lab(nr)
    return c["taburi"][tab] if c else None


def fisa_md(nr):
    """Fișa unui laborator ca text Markdown (pentru ecran și pentru descărcare)."""
    c = lab(nr)
    if not c:
        return f"## Laborator {nr} · {NUME_LAB[nr]}\n\n*Notebook-ul nu a fost rulat încă.*\n"
    f = c["fisa"]
    r = [f"## Laborator {nr} · {c['titlu']}", "", f"**Întrebare:** {c['intrebare']}", "", "**Cifre-cheie**", ""]
    r += [f"- {e[0]}: **{e[1]}**" for e in c["kpi"]]
    r += ["", "**Ce am făcut**", ""] + [f"{i}. {t}" for i, t in enumerate(f["pe_scurt"], 1)]
    de_ce = [t for k in (4, 5, 3) for t in c["taburi"][k]["de_ce"][:1]]
    r += ["", "**De ce așa**", ""] + [f"- {t}" for t in de_ce]
    intrebari = [q for k in (4, 5, 3) for q in c["taburi"][k]["intrebari"][:1]]
    r += ["", "**Întrebări posibile**", ""] + [f"- *{q}* {a}" for q, a in intrebari]
    r += ["", "**Limite**", ""] + [f"- {t}" for t in f["limite"]]
    r += ["", f"**Concluzie:** {c['concluzie']}", "", "**Text de vorbit (aprox. 40 de secunde)**", "", f"> {c['text_vorbit']}", ""]
    return "\n".join(r)


def toate_fisele_md():
    return "# Fișe de prezentare · Machine Learning\n\n" + "\n---\n\n".join(fisa_md(i) for i in range(1, 7))
