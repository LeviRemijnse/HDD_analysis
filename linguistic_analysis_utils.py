import pandas as pd
import json
import pickle
import numpy as np
from .fficf_utils import frames_from_dict, split_on_space, create_output_folder, frames_collection
from .predicate_utils import get_predicate_vocabulary, predicates_from_dict
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

def titles_buckets_df(list_of_buckets):
    """extract titles and corresponding time buckets and put them in a list of lists"""
    list_of_lists = []

    for cluster in list_of_buckets:
        time_bucket = cluster[0]
        list_of_dicts = cluster[1]
        for doc in list_of_dicts:
            title = list(doc.keys())[0]
            title_bucket = [title, time_bucket]
            list_of_lists.append(title_bucket)
    df = pd.DataFrame(list_of_lists, columns=['Title', 'Time bucket'])
    return df

def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    """split a data frame into train, validate and test set, following preset percentages"""
    np.random.seed(seed)
    perm = np.random.permutation(df.index) #randomize DataFrame index
    m = len(df.index) #get length of index
    train_end = int(train_percent * m) #calculate size of training set
    validate_end = int(validate_percent * m) + train_end #calculate size of validation set
    train = df.loc[perm[:train_end]] #create new DataFrame from the randomized index of training set size
    validate = df.loc[perm[train_end:validate_end]] #create new DataFrame from the randomized index of validate set size
    test = df.loc[perm[validate_end:]] #create new DataFrame from the remaining index
    return train, validate, test

def split_df(df):
    """split the dataframe into training, development and test dataframes"""
    time_buckets = df["Time bucket"].unique() #get unique labels from DataFrame
    grouped_df = df.groupby("Time bucket") #group the DataFrame by time bucket labels

    train_dfs = []
    develop_dfs = []
    test_dfs = []

    for time_bucket in time_buckets: #iterate over labels
        group = grouped_df.get_group(time_bucket) #create new DataFrame from each subset with different label
        train, develop, test = train_validate_test_split(group) #split Dataframe into train, develop, test
        assert len(train) > len(develop), "training set not larger than development set"
        assert len(train) > len(test), "training set not larger than test set"
        train_dfs.append(train) #append training DataFrame to list
        develop_dfs.append(develop) #append dev DataFrame to list
        test_dfs.append(test) #append test DataFrame to list

    assert len(train_dfs) == len(time_buckets), "not all time bucket labels have their own set"

    train_df = pd.concat(train_dfs) #concatenate training DataFrames
    develop_df = pd.concat(develop_dfs) #concatenate dev DataFrames
    test_df = pd.concat(test_dfs) #concatenate test DataFrames
    assert len(train_df) > len(develop_df), "training set not larger than development set"
    assert len(train_df) > len(test_df), "training set not larger than test set"
    return train_df, develop_df, test_df

def df_to_pickle(train_df, dev_df, test_df, train_path, dev_path, test_path, output_folder, start_from_scratch, verbose):
    """export DataFrames to pickle"""
    if output_folder != None:
        create_output_folder(output_folder=output_folder,
                            start_from_scratch=start_from_scratch,
                            verbose=verbose)
    train_df.to_pickle(train_path)
    dev_df.to_pickle(dev_path)
    test_df.to_pickle(test_path)

def discourse_ratio(doc, frames):
    """get the discourse sensitive ratio per frame for one document"""
    frames = set(frames)
    discourse_ratio_dict = {}

    for title, info in doc.items(): #iterate over title:info of document
        for frame in frames: #iterate over set of frames
            total = 0
        for term_id, frame_info in info['frame info'].items(): #iterate over term id:frame info
            if frame_info['frame'] == frame:
                sent_id = int(frame_info['sentence']) #get sentence number
                weighted_frame_occ = 1/sent_id #get discourse ratio
                total += weighted_frame_occ #add discourse ratio for each token of the same frame in doc
            discourse_ratio_dict[frame] = total #add frame:total discourse ratio to dict
    return discourse_ratio_dict

def get_frame_vocabulary(historical_distance_dict,event_type):
    """extract frames for an event type and put them in a list"""
    list_of_lists = []

    for cluster in historical_distance_dict[event_type]:
        day = cluster[0]
        collection = cluster[1]
        frames = frames_collection(collection)
        list_of_lists.append(frames)

    flat_list = [item for sublist in list_of_lists for item in sublist]
    vocabulary = list(set(flat_list))
    return vocabulary

def bag_of_predicates(historical_distance_info_dict, event_type, titles_df, verbose):
    """create BOW features for all lexical units across documents"""
    vocabulary = get_predicate_vocabulary(historical_distance_dict=historical_distance_info_dict,
                                            event_type=event_type,
                                            verbose=verbose)
    target_titles = set(titles_df["Title"])
    input_vec = []
    time_bucket_list = []
    titles = []

    for tupl in historical_distance_info_dict[event_type]:
        time_bucket = tupl[0]
        list_of_dicts = tupl[1]
        for doc in list_of_dicts:
            title = list(doc.keys())[0]
            if title not in target_titles:
                continue
            predicates = frames_from_dict(doc) #extract list of frames from doc
            space = ' '
            space = space.join(predicates) #join the frames
            input_vec.append(space) #append string of frames to list
            time_bucket_list.append(time_bucket) #append time bucket to time buckets list
            titles.append(title) #append title of doc to list

    vectorizer = CountVectorizer(analyzer=split_on_space, min_df=2, vocabulary=vocabulary) #initiate vectorizer
    predicates_vector_data = vectorizer.fit_transform(input_vec) #fit transform input
    headers = vectorizer.get_feature_names() #get frames as column headers
    predicates_vector_array = predicates_vector_data.toarray() #output vectorizer to array

    list_of_lists = []

    for doc, bucket, title in zip(predicates_vector_array, time_bucket_list, titles): #iterate over both vectors, time buckets and titles
        one_row = []
        for count, predicate in zip(doc, headers): #iterate over both values and corresponding frames
            one_row.append(count) #append value to row
        one_row.append(bucket) #append time bucket as final value to row
        list_of_lists.append(one_row) #append row to list

    headers.append("time bucket") #append time bucket header to headers list

    assert len(input_vec) == len(list_of_lists), "not all documents are represented"
    for row in list_of_lists:
        assert len(row) == len(headers), "not all frames are represented in row"
        assert type(row[-1]) == str, "time bucket not in row"

    df = pd.DataFrame(list_of_lists, columns=headers) #create DataFrame
    return df

def doc_features(historical_distance_info_dict, vocabulary, titles_df, event_type, selected_features, discourse_sensitive, path_typicality_scores, use_typicality_scores, verbose):
    """get features on document level"""
    target_titles = set(titles_df["Title"])
    input_vec = []
    time_bucket_list = []
    titles = []
    frames_discourse_ratios_dict = {}

    if use_typicality_scores:
        with open(path_typicality_scores, "r") as infile:
            typicality_scores_dict = json.load(infile) #load typicality scores

    for tupl in historical_distance_info_dict[event_type]: #iterate over clusters in event type
        time_bucket = tupl[0]
        list_of_dicts = tupl[1]
        for doc in list_of_dicts: #iterate over docs in cluster
            title = list(doc.keys())[0]
            if title not in target_titles:
                continue
            frames = frames_from_dict(doc) #extract list of frames from doc
            space = ' '
            space = space.join(frames) #join the frames
            input_vec.append(space) #append string of frames to list
            time_bucket_list.append(time_bucket) #append time bucket to time buckets list
            titles.append(title) #append title of doc to list
            if discourse_sensitive == True:
                discourse_ratio_dict = discourse_ratio(doc, frames) #get dict of discourse sensitive ratios per frame for doc
                frames_discourse_ratios_dict[title] = discourse_ratio_dict #add doc title:discourse ratios to dict

    vectorizer = CountVectorizer(lowercase=False, analyzer=split_on_space, vocabulary=vocabulary) #initiate vectorizer
    frames_vector_data = vectorizer.fit_transform(input_vec) #fit transform input
    headers = vectorizer.get_feature_names() #get frames as column headers
    frames_vector_array = frames_vector_data.toarray() #output vectorizer to array

    list_of_lists = []

    for doc, bucket, title in zip(frames_vector_array, time_bucket_list, titles): #iterate over both vectors, time buckets and titles
        one_row = []
        for count, frame in zip(doc, headers): #iterate over both values and corresponding frames
            freq = count
            if discourse_sensitive == True and frame in frames_discourse_ratios_dict[title]:
                ratio = frames_discourse_ratios_dict[title][frame] #get discourse ratio for frame
                freq = freq*ratio #modify value
                assert freq != 0, "feature modification failed"
            if use_typicality_scores:
                if frame in typicality_scores_dict[event_type]:
                    score = typicality_scores_dict[event_type][frame] #get typicality score
                    freq = freq*score #modify value
            one_row.append(freq) #append value to row
        one_row.append(bucket) #append time bucket as final value to row
        list_of_lists.append(one_row) #append row to list

    headers.append("time bucket") #append time bucket header to headers list

    assert len(input_vec) == len(list_of_lists), "not all documents are represented"
    for row in list_of_lists:
        assert len(row) == len(headers), "not all frames are represented in row"
        assert type(row[-1]) == str, "time bucket not in row"

    df = pd.DataFrame(list_of_lists, columns=headers) #create DataFrame
    return df

def extract_features_and_labels(df):
    """extract features and labels from a data frame"""
    labels = []

    for row in df.itertuples(): #iterate over rows in DataFrame
        labels.append(row[-1]) #append time bucket label to list

    del df["time bucket"] #delete label column from DataFrame
    data = df.T.to_dict().values() #convert DataFrame to list with dict per row

    assert len(data) == len(labels), "not all data or labels represented"
    return data, labels

def train_classifier(train_features, train_targets):
    """train the linear SVM classifier"""
    #model = SVC(kernel="rbf") #initiate SVM
    model = LinearSVC(dual=False)
    vec = DictVectorizer() #initiate vectorizer
    features_vectorized = vec.fit_transform(train_features) #vectorize features
    model.fit(features_vectorized, train_targets) #train classifier
    return model, vec

def classify_data(model, vec, features):
    """classify data with linear SVM classifier"""
    vec_features = vec.transform(features) #vectorize features
    predictions = model.predict(vec_features) #run classifier
    return predictions

def evaluation(human_annotation, system_output,report_path):
    """show evaluation metrics of multi-class system output in one classification report"""
    report = classification_report(human_annotation, system_output, digits=3, output_dict=True) #create report
    accuracy = f"accuracy {report['accuracy']}"
    del report['accuracy']
    df = pd.DataFrame.from_dict(report, orient='index')
    df = df.round(3)
    df_latex = df.to_latex()

    with open(report_path, 'w') as outfile:
        outfile.write(df_latex)

    with open(report_path, "a") as outfile:
        outfile.write(accuracy)
    return df

def remove_columns_with_zeros(train_df, dev_df, test_df, verbose=0):

    if verbose >= 5:
        print()
        print(f'length train_df: {len(train_df)}')
        print(f'length dev_df: {len(dev_df)}')
        print(f'length test_df: {len(test_df)}')

    big_df = pd.concat([train_df,
                            dev_df,
                            test_df], axis=0)

    assert (len(train_df) + len(dev_df) + len(test_df)) == len(big_df)

    if verbose >= 5:
        print()
        print(f'length joined df: {len(big_df)}')

    to_remove = []
    for column in big_df.columns:
        values = set(big_df[column])
        if values == {0}:
            to_remove.append(column)

    if verbose >= 5:
        print(f'{len(to_remove)} frames have zeros for all rows of all dataframes')


    train_df = train_df.drop(columns=to_remove)
    dev_df = dev_df.drop(columns=to_remove)
    test_df = test_df.drop(columns=to_remove)

    assert list(train_df.columns) == list(dev_df.columns)
    assert list(train_df.columns) == list(test_df.columns)
    assert list(dev_df.columns) == list(test_df.columns)

    if verbose >= 5:
        print()
        print()
        print(f'# of columns train_df: {len(train_df.columns)}')
        print(f'# of columns dev_df: {len(dev_df.columns)}')
        print(f'# of columns test_df: {len(test_df.columns)}')

    return train_df, dev_df, test_df
