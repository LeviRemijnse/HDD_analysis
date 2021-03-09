import sys
import os
sys.path.append('../../')

from HDD_analysis import event_type_info, linguistic_analysis, dir_path

typicality_scores = f"{dir_path}/test/output/c_tf_idf.json"
absolute_path = f"{dir_path}/test/output"
#time_buckets = {"day_0":range(0,1), "day_1":range(1,2), "day_2_to_30":range(2,31), "day_31_beyond":range(31,100000000)}
time_buckets = {"day_0":range(0,1), "day_3-30":range(3,31), "day_365_beyond":range(365,100000000)}
selected_features = ['frame frequency', 'frame root', 'nominalization', 'definiteness']

linguistic_analysis(time_bucket_config=time_buckets,
                    absolute_path=absolute_path,
                    report_filename="frequency_report",
                    project='GVA',
                    language='en',
                    event_type="Q5618454",
                    selected_features=selected_features,
                    discourse_sensitive=False,
                    typicality_scores=None,
                    verbose=5)
