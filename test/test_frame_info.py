import sys
import os
sys.path.append('../../')

from HDD_analysis import dir_path, frame_info

path = f'{dir_path}/test/input_files/VJ Cozer - Creator Entertainment.naf'

frame_info_dict = frame_info(naf_root=path,
                                verbose=2)
