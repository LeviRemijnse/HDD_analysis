import sys
import os
import json
sys.path.append('../../')

from HDD_analysis import dir_path, prepare_odd_one_out

fficf_path = f'{dir_path}/test/output/ff_icf.xlsx'
range_dict = {"high": (1.0, 10.0), "middle": (0.5, 1.0),"low": (0, 0.5)}
range_dict2 = {"high": (0.5, 10.0), "middle": (0.1, 0.5),"low": (0, 0.1)}
roots = ["Attributes"]

prepare_odd_one_out(frames_path=fficf_path,
                    event_type="Q5618454",
                    range_dict=range_dict2,
                    roots=roots,
                    item_length=3,
                    use_shortest_path=False,
                    use_shortest_path_to_root=True,
                    verbose=2)
