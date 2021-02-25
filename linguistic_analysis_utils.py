import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from collections import Counter
from nltk.corpus import framenet as fn
from .historical_distance_utils import historical_distance_frames
from .frame_relation_utils import get_inheritance_relations, get_digraph, get_frame_to_root_information
from .fficf_utils import frames_from_dict, split_on_space
from sklearn.feature_extraction.text import CountVectorizer

def sample_time_buckets(historical_distance_info_dict, event_type):
    """
    create proportional sizes of subcorpora across time buckets. randomize when selecting the texts for this sample.
    :param historical_distance_info_dict: collection of collections of dictionaries per event type
    :param event_type:
    :type historical_distance_info_dict: dictionary
    :type event_type:
    """
    lengths_dict = {}

    for cluster in historical_distance_info_dict[event_type]:
        time_bucket = cluster[0]
        list_of_dicts = cluster[1]
        lengths_dict[time_bucket] = len(list_of_dicts)

    len_smallest_corpus = min(lengths_dict.values())
    sampled_collections = {}

    for event_type, info_dicts in collections.items():
        sampled_list = random.sample(info_dicts, len_smallest_corpus)
        sampled_collections[event_type] = sampled_list

    return sampled_collections

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

def nominalization(historical_distance_info_dict, event_type, features):
    """for each frame get the distribution of nouns and verbs per time bucket"""
    for feature in features:
        name = feature['frame']
        nominalization_dict = {}
        for tupl in historical_distance_info_dict[event_type]:
            noun_count = 0
            verb_count = 0
            time_bucket = tupl[0]
            list_of_dicts = tupl[1]
            for info_dict in list_of_dicts:
                for title, info in info_dict.items():
                    for term_id, frame_info in info['frame info'].items():
                        if frame_info['frame'] == name:
                            if frame_info['POS'] == 'NOUN':
                                noun_count += 1
                                break
                            elif frame_info['POS'] == 'VERB':
                                verb_count += 1
                                break
                            else:
                                break
            if noun_count+verb_count == 0:
                ratio_dict = {'verb': verb_count, 'noun': noun_count}
            else:
                ratio_verbs = verb_count/(verb_count+noun_count)
                ratio_nouns = noun_count/(verb_count+noun_count)
                assert ratio_verbs+ratio_nouns == 1, "ratio verbs or nouns incorrect"
                ratio_dict = {'verb': ratio_verbs, 'noun': ratio_nouns}
            nominalization_dict[time_bucket] = ratio_dict
        feature['POS'] = nominalization_dict
    return

def definiteness(historical_distance_info_dict, event_type, features):
    """get definiteness ratio of nouns per frame per time bucket"""
    for feature in features:
        name = feature['frame']
        definiteness_dict = {}
        for tupl in historical_distance_info_dict[event_type]:
            definite_articles = 0
            indefinite_articles = 0
            time_bucket = tupl[0]
            list_of_dicts = tupl[1]
            for info_dict in list_of_dicts:
                for title, info in info_dict.items():
                    for term_id, frame_info in info['frame info'].items():
                        if frame_info['frame'] == name:
                            if frame_info['POS'] == 'NOUN' and frame_info['article']['definite'] == True:
                                definite_articles += 1
                                break
                            elif frame_info['POS'] == 'NOUN' and frame_info['article']['definite'] == False:
                                indefinite_articles += 1
                                break
                            else:
                                break
            if definite_articles+indefinite_articles == 0:
                definiteness_dict[time_bucket] = 0.0
            else:
                ratio_definiteness = definite_articles/(definite_articles+indefinite_articles)
                definiteness_dict[time_bucket] = ratio_definiteness
        feature['definiteness ratio'] = definiteness_dict

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
        if feature == "nominalization":
            nominalization(historical_distance_info_dict, event_type, features)
        if feature == "definiteness":
            definiteness(historical_distance_info_dict, event_type, features)
    print(features)
    return features

def discourse_ratio(doc, frames):
    """get the discourse sensitive ratio per frame for one document"""
    frames = set(frames)
    discourse_ratio_dict = {}

    for title, info in doc.items():
        for frame in frames:
            total = 0
        for term_id, frame_info in info['frame info'].items():
            if frame_info['frame'] == frame:
                sent_id = int(frame_info['sentence'])
                weighted_frame_occ = 1/sent_id
                total += weighted_frame_occ
            discourse_ratio_dict[frame] = total
    return discourse_ratio_dict

def doc_features(historical_distance_info_dict, event_type, selected_features, discourse_sensitive, typicality_scores, verbose):
    """get features on document level"""
    input_vec = []
    time_bucket_list = []
    titles = []
    frames_discourse_ratios_dict = {}

    if typicality_scores != None:
        with open(typicality_scores, "r") as infile:
            typicality_scores_dict = json.load(infile)

    for tupl in historical_distance_info_dict[event_type]:
        time_bucket = tupl[0]
        list_of_dicts = tupl[1]
        for doc in list_of_dicts:
            frames = frames_from_dict(doc)
            space = ' '
            space = space.join(frames) #join the frames
            input_vec.append(space)
            time_bucket_list.append(time_bucket)
            for title in doc:
                titles.append(title)
            if discourse_sensitive == True:
                discourse_ratio_dict = discourse_ratio(doc, frames)
                frames_discourse_ratios_dict[title] = discourse_ratio_dict

    vectorizer = CountVectorizer(lowercase=False, analyzer=split_on_space)
    frames_vector_data = vectorizer.fit_transform(input_vec)
    headers = vectorizer.get_feature_names()
    frames_vector_array = frames_vector_data.toarray()

    list_of_lists = []

    for doc, bucket, title in zip(frames_vector_array, time_bucket_list, titles):
        one_row = []
        for count, frame in zip(doc, headers):
            freq = count
            if discourse_sensitive == True and frame in frames_discourse_ratios_dict[title]:
                ratio = frames_discourse_ratios_dict[title][frame]
                freq = freq*ratio
                assert freq != 0, "feature modification failed"
            if typicality_scores != None and frame in typicality_scores_dict[event_type]:
                score = typicality_scores_dict[event_type][frame]
                freq = freq*score
            one_row.append(freq)
        one_row.append(bucket)
        list_of_lists.append(one_row)

    headers.append("time bucket")

    assert len(input_vec) == len(list_of_lists), "not all documents are represented"
    for row in list_of_lists:
        assert len(row) == len(headers), "not all frames are represented in row"
        assert type(row[-1]) == str, "time bucket not in row"

    df = pd.DataFrame(list_of_lists, columns=headers)
    return df

def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test

def split_df(df):
    """split the dataframe into training, development and test dataframes"""
    time_buckets = df["time bucket"].unique()
    grouped_df = df.groupby("time bucket")

    train_dfs = []
    develop_dfs = []
    test_dfs = []

    for time_bucket in time_buckets:
        group = grouped_df.get_group(time_bucket)
        train, develop, test = train_validate_test_split(group)
        assert len(train) > len(develop), "training set not larger than development set"
        assert len(train) > len(test), "training set not larger than test set"
        train_dfs.append(train)
        develop_dfs.append(develop)
        test_dfs.append(test)

    assert len(train_dfs) == len(time_buckets), "not all time bucket labels have their own set"

    train_df = pd.concat(train_dfs)
    develop_df = pd.concat(develop_dfs)
    test_df = pd.concat(test_dfs)
    assert len(train_df) > len(develop_df), "training set not larger than development set"
    assert len(train_df) > len(test_df), "training set not larger than test set"
    return train_df, develop_df, test_df
