import sys
import os
sys.path.append('../../')

from HDD_analysis import get_paths

paths = get_paths(project='HistoricalDistanceData',
                    language='en',
                    verbose=2)
