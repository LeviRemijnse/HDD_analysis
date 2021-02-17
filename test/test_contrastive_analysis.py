import sys
import os
sys.path.append('../../')

from HDD_analysis import get_naf_paths, event_type_info, fficf_info

output_folder = 'output'
xlsx_paths = ['output/tf_idf_means_per_event_type_no_cutoff.xlsx', 'output/c_tf_idf_no_cutoff.xlsx']
analysis_types = ["tf_idf", "c_tf_idf"]
generic_frames = ['Statement', 'Calendric_unit', 'Locative_relation', 'Cardinal_numbers','People']

event_type_frames_dict = fficf_info(project='HistoricalDistanceData',
                                    language='en',
                                    analysis_types=analysis_types,
                                    xlsx_paths=xlsx_paths,
                                    output_folder=output_folder,
                                    start_from_scratch=False,
                                    stopframes=generic_frames,
                                    verbose=3)
