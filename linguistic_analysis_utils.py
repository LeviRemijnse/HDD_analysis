import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import framenet as fn
from .historical_distance_utils import historical_distance_frames
from .frame_relation_utils import get_inheritance_relations, get_digraph, get_frame_to_root_information

def visualize_frequency(historical_distance_frames_dict,event_type,frames,pdf_path,verbose):
    """extract frequencies of specified frames per time bucket and visualize"""
    headers = ["time bucket", "frame", "frequency"]
    list_of_lists = []

    for frame in frames:
        for tupl in historical_distance_frames_dict[event_type]:
            one_row = []
            time_bucket = tupl[0]
            one_row.append(time_bucket)
            clustered_frames = tupl[1]
            frequencies = Counter(clustered_frames)
            one_row.append(frame)
            one_row.append(frequencies[frame])
            list_of_lists.append(one_row)

    df = pd.DataFrame(list_of_lists, columns=headers)
    lineplot = sns.lineplot(data=df, x="time bucket", y="frequency", hue="frame")
    plt.savefig(pdf_path)

    return

def frame_frequency(frames_dict, historical_distance_info_dict, event_type, features, discourse_sensitive=True):
    """extracts frequency per frame per time bucket and return as feature"""
    count_list = []

    for tupl in frames_dict[event_type]:
        time_bucket = tupl[0]
        clustered_frames = tupl[1]
        frequencies = Counter(clustered_frames)
        new_tuple = (time_bucket, frequencies)
        count_list.append(new_tuple)

    for feature in features:
        name = feature['frame']
        frequencies = {}
        for tupl in count_list:
            time_bucket = tupl[0]
            counted_frames = tupl[1]
            if name in counted_frames:
                frequencies[time_bucket] = counted_frames[name]
            else:
                frequencies[time_bucket] = 0
        feature['frequency'] = frequencies

    if discourse_sensitive == True:
        for feature in features:
            name = feature['frame']
            sensitive_frequencies = {}
            for tupl in historical_distance_info_dict[event_type]:
                total = 0
                time_bucket = tupl[0]
                list_of_dicts = tupl[1]
                for info_dict in list_of_dicts:
                    for title, info in info_dict.items():
                        for term_id, frame_info in info['frame info'].items():
                            if frame_info['frame'] == name:
                                sent_id = int(frame_info['sentence'])
                                weighted_frame_occ = 1/sent_id
                                total += weighted_frame_occ
                sensitive_frequencies[time_bucket] = total
            feature['discourse sensitive ratio'] = sensitive_frequencies
    return

def frame_root(fn, features, nodes=True):
    """for each frame get the root frame through inheritance relations"""
    sub_to_super = get_inheritance_relations(fn=fn, verbose=1)
    g, roots = get_digraph(sub_to_super, verbose=1)
    frame_to_root_info = get_frame_to_root_information(di_g=g, fn=fn, roots=roots, verbose=1)

    for feature in features:
        name = feature['frame']
        root_info = frame_to_root_info[name]
        root = root_info['root']
        feature['root'] = root
        if nodes == True:
            n_nodes = root_info['len_path']
            feature['number of nodes'] = n_nodes

def get_features(historical_distance_info_dict, event_type, selected_features, verbose):
    """extract features from corpus"""
    frames_dict = historical_distance_frames(historical_distance_info_dict)

    frames = set()

    for tupl in frames_dict[event_type]:
        time_bucket = tupl[0]
        clustered_frames = tupl[1]
        for frame in clustered_frames:
            frames.add(frame)

    features = []

    for frame in frames:
        frame_dict = {"frame": frame}
        features.append(frame_dict)

    for feature in selected_features:
        if feature == "frame frequency":
            frame_frequency(frames_dict, historical_distance_info_dict, event_type, features)
        if feature == "frame root":
            frame_root(fn, features)

    print(features)
    return features
