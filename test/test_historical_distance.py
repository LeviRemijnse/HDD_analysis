import sys
import os
sys.path.append('../../')

from HDD_analysis import historical_distance

output_folder = 'output/time_bucket_configuration_1'
json_path = 'output/time_bucket_configuration_1/sampled_corpus.json'
time_buckets = {"day_0":range(0,1), "day_1":range(1,2), "day_2_to_30":range(2,31), "day_31_beyond":range(31,100000000)}

historical_distance_info_dict = historical_distance(project='HistoricalDistanceData',
                                                    language='en',
                                                    output_folder=output_folder,
                                                    json_path=json_path,
                                                    dct_time_buckets=time_buckets,
                                                    verbose=3)
