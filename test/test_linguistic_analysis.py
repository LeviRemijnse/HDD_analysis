import sys
import os
sys.path.append('../../')

from HDD_analysis import linguistic_analysis, dir_path

absolute_path = f"{dir_path}/test/output"
time_buckets = {"day_0":range(0,1), "day_8-30":range(7,31)}

linguistic_analysis(time_bucket_config=time_buckets,
                        absolute_path=absolute_path,
                        experiment="frequency",
                        project='GVA',
                        language='en',
                        event_type="Q5618454",
                        path_typicality_scores=None,
                        use_frames=True,
                        discourse_sensitive=False,
                        use_bow=False,
                        balanced_classes=True,
                        verbose=5)
