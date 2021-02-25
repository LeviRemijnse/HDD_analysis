import sys
import os
sys.path.append('../../')

from HDD_analysis import historical_distance, linguistic_analysis

output_folder = '../test/output'
pdf_path = '../test/output/frequency.pdf'
typicality_scores = "../test/output/c_tf_idf.json"
frames = ['Attack','Weapon','Law_enforcement_agency', 'Leadership', 'Calendric_unit', 'Statement']
time_buckets = {"day_0":range(0,1), "day_1":range(1,2), "day_2_to_30":range(2,31), "day_31_beyond":range(31,100000000)}
selected_features = ['frame frequency', 'frame root', 'nominalization', 'definiteness']



historical_distance_info_dict = historical_distance(project='HistoricalDistanceData',
                                                    language='en',
                                                    dct_time_buckets=time_buckets,
                                                    verbose=3)

linguistic_analysis(historical_distance_info_dict=historical_distance_info_dict,
                    event_type="Q750215",
                    selected_features=selected_features,
                    typicality_scores=typicality_scores,
                    verbose=1)
