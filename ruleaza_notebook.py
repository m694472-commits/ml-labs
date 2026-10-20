"""Rulează un notebook de pe disc, de la cap la coadă, fără VS Code.
Folosire (din ML_labs):  python ruleaza_notebook.py laborator_1_2_regresie_clasificare.ipynb
Rezultatul (cu ieșiri) se salvează în același fișier; artefactele ajung în `artefacte/`.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

if len(sys.argv) < 2:
    sys.exit("Folosire: python ruleaza_notebook.py <notebook.ipynb>")
cale = Path(sys.argv[1]).resolve()
nb = nbformat.read(cale, as_version=4)
total = sum(c.cell_type == "code" for c in nb.cells)
print(f"Rulez {cale.name} ({total} celule de cod) din {cale.parent} ...", flush=True)
t0 = time.time()
client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": str(cale.parent)}})
gata = [0]


def dupa_celula(cell, cell_index, **kw):
    if cell.cell_type == "code":
        gata[0] += 1
        print(f"  celula {gata[0]}/{total}  ({time.time() - t0:.0f} s)", flush=True)


client.on_cell_executed = dupa_celula
try:
    client.execute()
except Exception as e:
    print("\nEROARE:", str(e)[-1500:])
    nbformat.write(nb, cale)
    sys.exit(1)
nbformat.write(nb, cale)
print(f"\nGATA în {time.time() - t0:.0f} s. Rulează acum: python verifica_artefacte.py")
