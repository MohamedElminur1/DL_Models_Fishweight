"""Generate Table 9, the descriptive T1 time-only all-events comparison."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase4 import result_table
if __name__ == '__main__':
    result_table('all_events', 'lopo', 'all', ('all_sample_set',), compact=True)
