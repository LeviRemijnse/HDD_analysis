import sys
import os
sys.path.append('../../')

from HDD_analysis import get_paths, event_type_info, frame_predicate_distribution

event_type_paths_dict = get_paths('HistoricalDistanceData','en',verbose=2)
event_type_info_dict = event_type_info(collections=event_type_paths_dict)

output_folder = 'output'
xlsx_path = 'output/predicate_distribution.xlsx'

distribution_dataframe = frame_predicate_distribution(collections=event_type_info_dict,
                                                        xlsx_path=xlsx_path,
                                                        output_folder=output_folder,
                                                        start_from_scratch=False,
                                                        verbose=0)
