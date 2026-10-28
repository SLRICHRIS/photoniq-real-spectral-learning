"""CI: validated notebook and ensured embedded modules exactly match Git source."""
import sys
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent.parent
notebook = nbformat.read(ROOT/'notebooks/PhotonIQ_Real_Experimental_Training.ipynb', as_version=4)
nbformat.validate(notebook)
checked = 0
for cell in notebook.cells:
    if cell.cell_type != 'code' or not cell.source.startswith('%%writefile '):
        continue
    line, _, contents = cell.source.partition('\n')
    filename = Path(line.split('photoniq_runtime/')[-1])
    target = ROOT / 'src' / filename
    if not target.exists():
        raise AssertionError(f'Embedded notebook module is missing from repository: {filename}')
    if contents.rstrip() != target.read_text(encoding='utf-8').rstrip():
        raise AssertionError(f'Colab source diverges from GitHub file: {filename}')
    checked += 1
assert checked == 5, f'Expected exactly 5 embedded modules, got {checked}'
print(f'Notebook schema valid, {checked} modules identical to repository source.')
