"""Generate compact Table 4 and its full supplementary version, including sensitivities."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import GROUPS, result_table
if __name__ == '__main__':
    result_table('main_results_full', 'lopo', 'eligible', GROUPS)
    result_table('main_results', 'lopo', 'eligible', GROUPS[:-1], compact=True)
