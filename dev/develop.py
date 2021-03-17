import sys
import os
import json
sys.path.append('../../')

from HDD_analysis import dir_path, linguistic_analysis, frame_predicate_distribution

typicality_scores = f"{dir_path}/test/output/c_tf_idf.json"
absolute_path = f"{dir_path}/test/output"
time_buckets = {"day_0":range(0,1), "day_1":range(1,2), "day_2_to_30":range(2,31), "day_31_beyond":range(31,100000000)}
time_buckets2 = {"day_0":range(0,1), "day_3-30":range(3,31), "day_365_beyond":range(365,100000000)}
selected_features = ['frame frequency', 'frame root', 'nominalization', 'definiteness']
time_buckets3 = {"day_0":range(0,1), "day_8-365":range(7,366)}
time_buckets4 = {"day_0":range(0,1), "day_8-30":range(7,31)}

sampled_corpus = f"{dir_path}/test/output/day_0---day_8-30+balanced/sampled_corpus.json"
with open(sampled_corpus, "r") as infile:
    sampled_corpus = json.load(infile)

time_bucket_predicate_dict = {}
xlsx_path = f"{dir_path}/test/output/day_0---day_8-30+balanced/predicate_distribution.xlsx"
for tupl in sampled_corpus["Q5618454"]:
    time_bucket = tupl[0]
    list_of_dicts = tupl[1]
    time_bucket_predicate_dict[time_bucket] = list_of_dicts

frame_predicate_distribution(collections=time_bucket_predicate_dict,
                             xlsx_path=xlsx_path,
                            output_folder=absolute_path,
                            start_from_scratch=False,
                            verbose=0)

#linguistic_analysis(time_bucket_config=time_buckets4,
                #        absolute_path=absolute_path,
                #        experiment="bow",
                #        project='GVA',
                #        language='en',
                #        event_type="Q5618454",
                #        selected_features=selected_features,
                #        path_typicality_scores=None,
                #        use_frames=False,
                #        discourse_sensitive=False,
                #        use_bow=True,
                #        balanced_classes=state,
                #        verbose=5)
