"""Generate Table 8, the descriptive calendar hold-out without intervals or tests."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import GROUPS, result_table
if __name__ == '__main__':
    result_table('calendar_holdout', 'chrono', 'eligible', GROUPS[:-1])
