import numpy as np
import operator
import os
import random
import statistics
import math
import json
import shutil
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def delete_smallest_texts(collections, minimal_n_frames,verbose):
    """
    load the event_type_info_dict and delete the smallest texts.
    :param collections: collection of collections of dictionaries per event type
    :param minimal_n_frames: filter of minimum number of annotated frames in a text
    :type collections: dictionary
    :type minimal_n_frames: integer
    """
    sliced_corpus = {}
    count = 0
    for event_type, events in collections.items():
        new_list = []
        for event in events:
            for title, stats in event.items():
                if stats['frame frequency'] >= minimal_n_frames:
                    new_list.append(event)
                else:
                    count += 1
                    continue
        assert len(new_list) != 0, "no documents containing more frames than provided threshold"
        sliced_corpus[event_type] = new_list
    if verbose >= 1:
        print(f"{count} texts with less than {minimal_n_frames} frames removed")
    return sliced_corpus

def sample_corpus(collections):
    """
    create proportional sizes of subcorpora across event types. randomize when selecting the texts for this sample.
    :param collections: collection of collections of dictionaries per event type
    :type collections: dictionary
    """
    lengths_dict = {}

    for event_type, info_dicts in collections.items():
        lengths_dict[event_type] = len(info_dicts)

    len_smallest_corpus = min(lengths_dict.values())
    sampled_collections = {}

    for event_type, info_dicts in collections.items():
        sampled_list = random.sample(info_dicts, len_smallest_corpus)
        sampled_collections[event_type] = sampled_list

    return sampled_collections

def frames_per_doc(frame_info_dict):
    """
    Load a naf dictionary, extract the frames and add them to a list.
    :param frame_info_dict: relevant linguistic information from a NAF file
    :type frame_info_dict: dictionary
    """
    title_frames_dict = {}
    frames = []

    for title, info_dict in frame_info_dict.items():
        for term_id, term_info in info_dict['frame info'].items():
            frame = term_info["frame"]
            frames.append(frame)
    title_frames_dict[title] = frames
    return title_frames_dict

def frames_per_doc_collection(collection):
    """
    returns a list for each NAF file a dictionary with title:list of frames.
    :param collection: collection of NAF files
    :type collection: list
    """

    collection_frames = []

    for info_dict in collection:
        title_frames_dict = frames_per_doc(info_dict)
        collection_frames.append(title_frames_dict)
    return collection_frames

def frames_per_doc_collections(event_type_frame_collections):
    """
    returns a dictionary with event type:list with dictionary (title:list of frames) per NAF file.
    :param event_type_frame_collections: collection of collections of event type with corresponding info-dicts for NAF files
    :type event_type_frame_collections: dictionary
    """
    event_type_frames_dict = {}

    for event_type, collection in event_type_frame_collections.items():
        event_type_frames_dict[event_type] = frames_per_doc_collection(collection)
    return event_type_frames_dict

def frames_from_dict(frame_info_dict):
    """
    Load a naf dictionary, extract the frames and add them to a list.
    :param frame_info_dict: dictionary with linguistic information extracted from NAF file
    :type frame_info_dict: dictionary
    """
    frames = []

    for title, info in frame_info_dict.items():
        for term_id, term_info in info['frame info'].items():
            frame = term_info["frame"]
            frames.append(frame)
    return frames

def frames_collection(collection):
    """
    returns a list of frames extracted from a collection of NAF files.
    :param collection: collection of dictionaries with relevant info extracted from NAF files
    :param collection: list
    """
    collection_frames = []

    for info_dict in collection:
        for frame in frames_from_dict(info_dict):
            collection_frames.append(frame)
    return collection_frames

def frames_collections(event_type_frame_collections):
    """
    returns a dictionary with the event type as key and list of frames as value
    :param event_type_frame_collections: collection of collections of event types with corresponding dictionaries for NAF files
    :type event_type_frame_collections: dictionary
    """
    event_type_frames_dict = {}

    for event_type, collection in event_type_frame_collections.items():
        event_type_frames_dict[event_type] = frames_collection(collection)
    return event_type_frames_dict

def frame_stats(event_type_frames_dict):
    """
    returns a dictionary with event type as key and a dictionary with (relative) frequency for each frame as value
    :param event_type_frames_dict: dictionary with for each event type a list of corresponding frames
    :type event_type_frame_dict: dictionary
    """

    event_type_frame_freq_dict = {}

    for key in event_type_frames_dict:
        assert type(event_type_frames_dict[key]) == list, "no list of frames"
        frame_stats_dict = {}
        frames = event_type_frames_dict[key]
        assert len(frames) != 0, "no frames in list"
        count_dict = Counter(frames)
        for frame, freq in count_dict.items():
            relative_freq = (freq/len(frames))*100
            info_dict = {'absolute frequency': freq, 'relative frequency': relative_freq}
            frame_stats_dict[frame] = info_dict
        event_type_frame_freq_dict[key] = frame_stats_dict
    return event_type_frame_freq_dict

def event_type_means(event_type_docs_tf_idfdict):
    """
    performs statistics to get tf_idf scores on event type level
    :param event_type_docs_tf_idfdict: dictionary with per event type and per document the tfidf scores
    :type event_type_docs_tf_idfdict: dictionary
    """
    event_type_tfidf_dict = {}

    for event_type, events in event_type_docs_tf_idfdict.items():
        frames = defaultdict(list)
        for event, frame_scores in events.items():
            for tupl in frame_scores:
                frame = tupl[0]
                score = tupl[1]
                frames[frame].append(score)
        event_type_tfidf_dict[event_type] = frames

    for event_type, frames in event_type_tfidf_dict.items():
        for frame, scores in frames.items():
            frames[frame] = statistics.mean(scores)
        ranked_scores = sorted(frames.items(), key=operator.itemgetter(1), reverse=True)
        event_type_tfidf_dict[event_type] = ranked_scores
    return event_type_tfidf_dict

def split_on_space(text):
    """
    split text on spaces.
    :param text: the text that needs to be tokenized
    :type text: string
    """
    return text.split(' ')

def tf_idf_doc_level(collections, stopframes, min_df):
    """
    calculate tf-idf scores on document level and get means per event type
    :param collections: collection of collections of event types with corresponding dictionaries per NAF file
    :param stopframes: list of frames for the stopwords parameter of the CountVectorizer
    :param min_df: minimal number of tokens of a frame that should be considered by the CountVectorizer
    :type collections: dictionary
    :type stopframes: list
    :type min_df: integer
    """
    event_type_frames_dict = frames_per_doc_collections(collections)

    incidents = []
    lists_frames = []

    for event_type, events in event_type_frames_dict.items():
        for info in events:
            for event, frames in info.items():
                values = frames #create a variable for each list of frames
                space = ' '
                space = space.join(values) #join the frames
                lists_frames.append(space)
                incidents.append(event)

    vectorizer = CountVectorizer(lowercase=False, stop_words=stopframes, analyzer=split_on_space, min_df=min_df) #frame vocabulary
    lists_vector_data = vectorizer.fit_transform(lists_frames) #data structure that represents the instances through their vectors
    column_headers = vectorizer.get_feature_names() #frame vocabulary mapped to data columns
    tfidf_transformer = TfidfTransformer()
    lists_frames_tfidf = tfidf_transformer.fit_transform(lists_vector_data)
    tf_idf_array = lists_frames_tfidf.toarray()
    tf_idf_array_round = np.round(tf_idf_array, decimals=3)

    tf_idfdict = {}

    for event, array in zip(incidents, tf_idf_array_round):
        frame_valuedict = {}
        for frame, value in zip(column_headers, array): #iterate over each frame and its corresponding value
            frame_valuedict[frame] = value #add the frame and value as key-value pair to a dictionary
            sorted_tuples = sorted(frame_valuedict.items(), key=operator.itemgetter(1), reverse=True) #convert the dictionary to a list of tuples sorted in descending order of the values
        tf_idfdict[event] = sorted_tuples #add the event type and its list of tuples as a key-value pair to the tf_idfdict

    event_type_docs_tf_idfdict = {}

    for event_type, events in event_type_frames_dict.items():
        events_dict = {}
        for info in events:
            for event, frames in info.items():
                for text, scores in tf_idfdict.items():
                    if event == text:
                        events_dict[text] = scores
        event_type_docs_tf_idfdict[event_type] = events_dict

    event_type_tfidf_dict = event_type_means(event_type_docs_tf_idfdict)
    return event_type_tfidf_dict

def c_tf_idf(frame_freq_event_type, total_freq_frames_event_type, total_n_docs, frame_freq_event_types):
    """
    calculates c_tf_idf for a frame in a given matrix.
    :param frame_freq_event_type: the number of occurences of the frame in the event type
    :param total_freq_frames_event_type: the total number of all frame occurrences in the event type
    :param total_n_docs: the total number of documents across event types
    :param frame_freq_event_types: the number of occurences of the frame across event types
    :type frame_freq_event_type: integer
    :type total_freq_frames_event_type: integer
    :type total_n_docs: integer
    :type frame_freq_event_types: integer
    """
    tf = frame_freq_event_type/total_freq_frames_event_type
    idf_array = np.log(np.divide(total_n_docs, frame_freq_event_types)).reshape(-1,1)
    idf = idf_array[0][0]
    c_tf_idf = tf*idf
    return c_tf_idf

def normalize_data(data):
    the_array = (data - np.min(data)) / (np.max(data) - np.min(data))
    return list(the_array)

def ff_icf(collections, event_type_frames_dict, frame_freq_dict):
    """
    calculates ff_icf scores.
    :param collections: collection of collections of event types with corresponding dictionaries with linguistc NAF info
    :param event_type_frames_dict: dictionary with event types: list of frames
    :param frame_freq_dict: absolute and relative frequency per frame per event type
    :type collections: dictionary
    :type event_type_frames_dict: dictionary
    :type frame_freq_dict: dictionary
    """
    total_n_docs = 0 #total number of documents

    for event_type, info in collections.items(): #iterate over event type: list of info-dictionaries
        total_n_docs += len(info) #add the length of the list (equal to the number of texts) to counter

    lists_frames = []

    for key in event_type_frames_dict: #iterate over the key:value (event type:list of frames) pairs
        values = event_type_frames_dict[key] #create a variable for each list of frames
        assert type(values) == list, "no list of frames"
        assert len(values) != 0, "no frames in list"
        space = ' '
        space = space.join(values) #join the frames
        lists_frames.append(space) #append the string to a list. this results in a list with a joined string of frames per event type

    vectorizer = CountVectorizer(lowercase=False, analyzer=split_on_space)
    frames_vector_data = vectorizer.fit_transform(lists_frames) #vectorize the frames per event type
    vector_shape = frames_vector_data.shape
    column_headers = vectorizer.get_feature_names() #get the matrix's columnh headers
    assert vector_shape[0] == len(collections), "not all event types are represented in matrix"
    assert vector_shape[1] == len(column_headers), "not all frames are represented in matrix"
    frames_vector_array = frames_vector_data.toarray()
    list_of_lists = []

    for row in frames_vector_array: #iterate over the vectors
        assert len(row) == len(column_headers), "certain frames in column headers not represented in vector"
        total_freq_frames_event_type = sum(row) #create variable for the sum of all frames of the event type
        scores = []
        for frame_freq_event_type, frame in zip(row, column_headers): #iterate over both the absolute frequencies in the vector and the corresponding column headers
            frame_freq_event_types = 0
            for event_type, freq_dict in frame_freq_dict.items(): #iterate over event_type:dictionary with frequencies
                if frame in freq_dict: #if the frame is in the dictionary of the event type
                    frame_freq_event_types += freq_dict[frame]['absolute frequency'] #add the absolute frequency to counter
            assert frame_freq_event_types != 0, f"{frame} not in corpus"
            c_tf_idf_score = c_tf_idf(frame_freq_event_type, total_freq_frames_event_type, total_n_docs, frame_freq_event_types)
            #if c_tf_idf_score < 0:
            #    c_tf_idf_score = 0
            scores.append(c_tf_idf_score) #append the score to a list
        normalized_scores = normalize_data(scores)
        list_of_lists.append(normalized_scores) #append the list to another list. The result is a list of scores per each event type

    c_tf_idf_round = np.round(list_of_lists, decimals=6)
    assert len(c_tf_idf_round) == len(collections), "not all event types are represented as list with c-tf-idf scores"
    c_tf_idfdict = {}

    for event_type, array in zip(event_type_frames_dict, c_tf_idf_round): #iterate over the event types and the list of corresponding arrays of tf-idf values
        frame_valuedict = {}
        for frame, value in zip(column_headers, array): #iterate over each frame and its corresponding value
            frame_valuedict[frame] = value #add the frame and value as key-value pair to a dictionary
        sorted_tuples = sorted(frame_valuedict.items(), key=operator.itemgetter(1), reverse=True) #convert the dictionary to a list of tuples sorted in descending order of the values
        assert len(sorted_tuples) == len(column_headers), "not all frames across event types have a c-tf-idf score in event type"
        c_tf_idfdict[event_type] = sorted_tuples #add the event type and its list of tuples as a key-value pair to the tf_idfdict
    return c_tf_idfdict

def contrastive_analysis(collections, analysis_types, stopframes, min_df, verbose):
    """returns a dictionary with event type as key and a sorted list of frames and their tf-idf values"""
    event_type_frames_dict = frames_collections(collections) #for each event type get a list of frames

    if verbose >= 2:
            for event_type, frames in event_type_frames_dict.items():
                print(f'{event_type}: {len(frames)} frames')
    frame_freq_dict = frame_stats(event_type_frames_dict)

    output_list = []

    for analysis_type in analysis_types:
        if analysis_type == "tf_idf":
            event_type_tfidf_dict = tf_idf_doc_level(collections, stopframes, min_df)

            if verbose >= 3:
                for event_type, scores in event_type_tfidf_dict.items():
                    print(f'{event_type}: top mean ranking: {scores[:3]}')
            output_list.append(event_type_tfidf_dict)

        if analysis_type == "c_tf_idf":
            c_tf_idf_dict = ff_icf(collections, event_type_frames_dict, frame_freq_dict)

            if verbose >= 3:
                for event_type, scores in c_tf_idf_dict.items():
                    print(f'{event_type}: top ranking: {scores[:3]}')
                    print(f'{event_type}: bottom ranking: {scores[-3:]}')
            output_list.append(c_tf_idf_dict)

    return output_list, frame_freq_dict

def create_output_folder(output_folder,start_from_scratch,verbose):
    '''creates output folder for export dataframe'''
    if os.path.isdir(output_folder):
        if start_from_scratch == True:
            shutil.rmtree(output_folder)
            if verbose >= 1:
                print(f"removed existing folder {output_folder}")

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        if verbose >= 1:
            print(f"created folder at {output_folder}")

def output_tfidf_to_format(tf_idfdict,frame_freq_dict,xlsx_path,output_folder,start_from_scratch,verbose):
    """exports the output of the tf-idf analysis to an excel format"""
    headers = ['event type', 'rank', 'frame', 'tf-idf value', 'absolute freq', 'relative freq', 'judgement']
    list_of_lists = []

    for key in tf_idfdict:
        cutoff_point = len(tf_idfdict[key])
        break

    for key in tf_idfdict:
        for tupl, number in zip(tf_idfdict[key][:cutoff_point], range(1,(cutoff_point+1))):
            one_row = [key, number]
            frame = tupl[0]
            score = tupl[1]
            one_row.append(frame)
            one_row.append(score)
            if frame in frame_freq_dict[key]:
                abs_freq = frame_freq_dict[key][frame]['absolute frequency']
                rel_freq = frame_freq_dict[key][frame]['relative frequency']
            else:
                abs_freq = 0
                rel_freq = 0
            one_row.append(abs_freq)
            one_row.append(rel_freq)
            one_row.append('')
            list_of_lists.append(one_row)

    df = pd.DataFrame(list_of_lists, columns=headers)

    if output_folder != None:
        create_output_folder(output_folder=output_folder,
                            start_from_scratch=start_from_scratch,
                            verbose=verbose)
        df.to_excel(xlsx_path, index=False)

    return df

def output_tfidf_to_json(tf_idfdict,json_path,output_folder,start_from_scratch,verbose):
    """exports the output of the tf-idf analysis to a json format"""
    json_dict = {}

    for key in tf_idfdict:
        scores_dict = {}
        for tupl in tf_idfdict[key]:
            frame = tupl[0]
            score = tupl[1]
            scores_dict[frame] = score
        json_dict[key] = scores_dict

    if output_folder != None:
        create_output_folder(output_folder=output_folder,
                            start_from_scratch=start_from_scratch,
                            verbose=verbose)
        with open(json_path, 'w') as outfile:
            json.dump(json_dict, outfile, indent=4, sort_keys=True)
    return

def top_n_frames(typicality_scores, top_n_typical_frames, verbose=0):
    if top_n_typical_frames == 'all':
        the_target_frames = list(typicality_scores)
    else:
        the_target_frames = []

        n = 0
        for frame_label, typ_score in sorted(typicality_scores.items(),
                                             key=operator.itemgetter(1),
                                             reverse=True):
            the_target_frames.append(frame_label)

            n += 1
            if n == top_n_typical_frames:
                break

        assert len(the_target_frames) == top_n_typical_frames

    if verbose >= 5:
        print()
        print(f'selected {len(the_target_frames)} from total of {len(typicality_scores)} frames')
        print(f'first three are: {the_target_frames[:3]}')

    return the_target_frames

def get_time_bucket_to_frame_to_freq(big_df,
                                     target_frames=None):

    the_time_buckets = set(big_df['time bucket'])
    time_bucket_to_frame_to_freq = {}
    frame_to_freq = defaultdict(int)
    time_bucket_to_total_frame_occurrences = defaultdict(int)

    for time_bucket in the_time_buckets:
        time_bucket_df = big_df[big_df['time bucket'] == time_bucket]
        time_bucket_to_frame_to_freq[time_bucket] = {}

        for frame_label in time_bucket_df.columns:
            if frame_label != 'time bucket':

                if target_frames:
                    if frame_label not in target_frames:
                        continue

                total = sum(time_bucket_df[frame_label])
                time_bucket_to_frame_to_freq[time_bucket][frame_label] = total
                frame_to_freq[frame_label] += total
                time_bucket_to_total_frame_occurrences[time_bucket] += total

    return time_bucket_to_frame_to_freq, frame_to_freq, time_bucket_to_total_frame_occurrences

def compute_c_tf_idf_between_time_buckets(typicality_scores,
                                          train_df,
                                          dev_df,
                                          test_df,
                                          top_n_typical_frames,
                                          verbose=0):
    the_target_frames = top_n_frames(typicality_scores=typicality_scores,
                                     top_n_typical_frames=top_n_typical_frames,
                                     verbose=verbose)

    the_big_df = pd.concat([train_df,
                                dev_df,
                                test_df], axis=0)


    # time_bucket -> frame -> frequency
    # total frequency of a frame across time buckets
    # total number of frame occurrences per time bucket
    time_bucket_to_frame_to_freq,\
    frame_to_freq, \
    time_bucket_to_total_frame_occurrences = get_time_bucket_to_frame_to_freq(big_df=the_big_df,
                                                                              target_frames=the_target_frames)


    # total number of docs
    number_of_docs = len(the_big_df)

    list_of_lists = []
    headers = ['time bucket', 'frame', 'c_tf_idf']

    for time_bucket, tb_frame_to_freq in time_bucket_to_frame_to_freq.items():

        for frame, tb_freq in tb_frame_to_freq.items():

            c_tf_idf_score = c_tf_idf(frame_freq_event_type=tb_freq,
                                      total_freq_frames_event_type=time_bucket_to_total_frame_occurrences[time_bucket],
                                      total_n_docs=number_of_docs,
                                      frame_freq_event_types=frame_to_freq[frame])
            one_row = [time_bucket, frame, c_tf_idf_score]
            list_of_lists.append(one_row)


    c_tf_idf_df = pd.DataFrame(list_of_lists, columns=headers)
    return c_tf_idf_df
