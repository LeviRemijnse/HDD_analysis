import sys
import os
sys.path.append('../')

import DFNDataReleases
from DFNDataReleases import get_relevant_info

relevant_info = get_relevant_info(repo_dir=DFNDataReleases.dir_path,
                                project='HistoricalDistanceData',
                                load_jsons=True)
