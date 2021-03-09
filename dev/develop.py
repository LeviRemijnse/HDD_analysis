import sys
import os
sys.path.append('../../')

from HDD_linguistic_analysis import dir_path, linguistic_analysis

typicality_scores = f"{dir_path}/test/output/c_tf_idf.json"
absolute_path = f"{dir_path}/test/output"
time_buckets = {"day_0":range(0,1), "day_1":range(1,2), "day_2_to_30":range(2,31), "day_31_beyond":range(31,100000000)}
time_buckets2 = {"day_0":range(0,1), "day_3-30":range(3,31), "day_365_beyond":range(365,100000000)}
selected_features = ['frame frequency', 'frame root', 'nominalization', 'definiteness']

time_buckets3 = {"day_0":range(0,1), "day_8-365":range(7,366)}

linguistic_analysis(time_bucket_config=time_buckets3,
                    absolute_path=absolute_path,
                    experiment="frequency",
                    project='GVA',
                    language='en',
                    event_type="Q5618454",
                    selected_features=selected_features,
                    use_frames=True,
                    discourse_sensitive=False,
                    typicality_scores=None,
                    use_bow=False,
                    balanced_classes=False,
                    verbose=5)

linguistic_analysis(time_bucket_config=time_buckets3,
                    absolute_path=absolute_path,
                    experiment="frequency_typicality",
                    project='GVA',
                    language='en',
                    event_type="Q5618454",
                    selected_features=selected_features,
                    use_frames=True,
                    discourse_sensitive=False,
                    typicality_scores=typicality_scores,
                    use_bow=False,
                    balanced_classes=False,
                    verbose=5)

experiments = {"frequency": {"use_frames": True,
                            'discourse_sensitive': False,
                            'typicality_scores': None,
                            'use_bow': False,
                            "balanced_classes": False},
                "frequency_typicality": {'use_frames': True,
                                        'discourse_sensitive': False,
                                        'typicality_scores': typicality_scores,
                                        'use_bow': False,
                                        "balanced_classes": False},
                "frequency_discoursesen": {'use_frames': True,
                                            'discourse_sensitive': True,
                                            'typicality_scores': None,
                                            'use_bow': False,
                                            "balanced_classes": False},
                "frequency_discoursesen_typicality": {'use_frames': True,
                                                        'discourse_sensitive': True,
                                                        'typicality_scores': typicality_scores,
                                                        'use_bow': False,
                                                        "balanced_classes": False},
                "bow": {'use_frames': False,
                        'discourse_sensitive': False,
                        'typicality_scores': None,
                        'use_bow': True,
                        "balanced_classes": False},
                "bow_frames": {'use_frames': True,
                                'discourse_sensitive': False,
                                'typicality_scores': None,
                                'use_bow': True,
                                "balanced_classes": False},
                "bow_frames_discoursesen": {'use_frames': True,
                                            'discourse_sensitive': True,
                                            'typicality_scores': None,
                                            'use_bow': True,
                                            "balanced_classes": False},
                "bow_frames_typicality": {"use_frames": True,
                                            'discourse_sensitive': False,
                                            'typicality_scores': typicality_scores,
                                            'use_bow': True,
                                            "balanced_classes": False},
                "bow_frames_discoursesen_typicality": {"use_frames": True,
                                                        'discourse_sensitive': True,
                                                        'typicality_scores': typicality_scores,
                                                        'use_bow': True,
                                                        "balanced_classes": False}}
