import sys
import os
sys.path.append('../../')

from HDD_analysis import frame_info

path = 'input_files'

collection = []

for root, directories, files in os.walk(path):
    for file in files:
        if file.endswith('.naf'):
            collection.append(os.path.join(root, file))

naf = collection[0]

frame_info_dict = frame_info(naf_root=naf,
                                verbose=2)
