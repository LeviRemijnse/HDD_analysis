import sys
import os
sys.path.append('../../')

from HDD_analysis import get_naf_paths, event_type_info, fficf_info, dir_path

output_folder = f'{dir_path}/test/output'
xlsx_paths = [f'{dir_path}/test/output/tf_idf_means_per_event_type.xlsx', f'{dir_path}/test/output/ff_icf.xlsx']
json_paths = [f'{dir_path}/test/output/tf_idf_means_per_event_type.json', f'{dir_path}/test/output/ff_icf.json']
gva_path = f'{dir_path}/test/output/unknown_distance.json'
json_path = f'{dir_path}/test/output/sampled_corpus.json'
analysis_types = ["tf_idf", "c_tf_idf"]
#generic_frames = ['Statement', 'Calendric_unit', 'Locative_relation', 'Cardinal_numbers','People']

event_type_frames_dict = fficf_info(project='HDD',
                                    language='en',
                                    analysis_types=analysis_types,
                                    xlsx_paths=xlsx_paths,
                                    output_folder=output_folder,
                                    start_from_scratch=False,
                                    json_paths=json_paths,
                                    gva_path=gva_path,
                                    json_path=json_path,
                                    verbose=3)
