import sys
import os
sys.path.append('../../')

from HDD_analysis import fficf_info, dir_path

output_folder = f'{dir_path}/test/output'
xlsx_paths = [f'{dir_path}/test/output/ff_icf.xlsx']
json_paths = [f'{dir_path}/test/output/ff_icf.json']
gva_path = f'{dir_path}/test/output/unknown_distance.json'
analysis_types = ["c_tf_idf"]

event_type_frames_dict = fficf_info(project='HDD',
                                    language='en',
                                    analysis_types=analysis_types,
                                    xlsx_paths=xlsx_paths,
                                    output_folder=output_folder,
                                    start_from_scratch=False,
                                    json_paths=json_paths,
                                    gva_path=gva_path,
                                    verbose=3)
