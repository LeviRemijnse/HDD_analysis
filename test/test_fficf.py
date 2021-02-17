import sys
import os
sys.path.append('../../')

from HDD_analysis import ff_icf, frame_stats

collections = {'Q1':
                  [{'text1':
                        ['Organization', 'Origin','People']},
                   {'text2':
                        ['Organization', 'Aggregate', 'Aggregate']}],
               'Q2':
                  [{'text3':
                        ['Organization', 'Origin','Create_physical_artwork']},
                   {'text4':
                        ['Organization', 'Origin', 'Type']}],
               'Q3':
                  [{'text5':
                        ['Organization', 'Intentionally_act','Custom']},
                   {'text6':
                        ['Organization', 'People', 'Similarity']}]}

event_type_frames_dict = {'Q1':
                            ['Organization', 'Origin','People', 'Organization', 'Aggregate', 'Aggregate'],
                          'Q2':
                            ['Organization', 'Origin','Create_physical_artwork', 'Organization', 'Origin', 'Type'],
                          'Q3':
                            ['Organization', 'Intentionally_act','Custom','Organization', 'People', 'Similarity']}

frame_freq_dict = frame_stats(event_type_frames_dict)
c_tf_idf_dict = ff_icf(collections, event_type_frames_dict, frame_freq_dict)
print("test complete")

for tupl in c_tf_idf_dict["Q1"]:
    frame = tupl[0]
    score = tupl[1]
    if frame == 'Organization':
        assert score == 0.0, f"{frame} gets {score} instead of 0.0" #0.0
    if frame == 'Origin':
        assert score == 0.115525, f"{frame} gets {score} instead of 0.115525"
    if frame == 'People':
        assert score == 0.183102, f"{frame} gets {score} instead of 0.183102"

max_score = c_tf_idf_dict["Q1"][0]
assert max_score[0] == "Aggregate", "highest scoring frame is not most typical"
