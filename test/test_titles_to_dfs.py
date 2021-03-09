import sys
import os
sys.path.append('../../')

from HDD_analysis import sampled_titles_to_dfs

output_folder = 'output/time_bucket_configuration_1'
sampled_corpus = "output/time_bucket_configuration_1/sampled_corpus.json"
train_path = "output/time_bucket_configuration_1/titles_train.pkl"
dev_path = "output/time_bucket_configuration_1/titles_dev.pkl"
test_path = "output/time_bucket_configuration_1/titles_test.pkl"

sampled_titles_to_dfs(historical_distance_info_dict=sampled_corpus,
                        event_type="Q750215",
                        train_path=train_path,
                        dev_path=dev_path,
                        test_path=test_path,
                        output_folder=output_folder,
                        verbose=3)
