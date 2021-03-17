from collections import defaultdict, Counter
import pandas as pd
import os

def predicates_from_dict(frame_info_dict):
    """
    Load a naf dictionary, extract the predicates and add them to a list.
    :param frame_info_dict: dictionary with linguistic information extracted from NAF file
    :type frame_info_dict: dictionary
    """
    predicates = []

    for title, info in frame_info_dict.items():
        for term_id, term_info in info['frame info'].items():
            lemma = term_info["lemma"]
            predicates.append(lemma)
    return predicates

def get_predicate_vocabulary(historical_distance_dict,event_type, verbose):
    """extract predicates for an event type and put them in a list"""
    predicates = []

    for cluster in historical_distance_dict[event_type]:
        time_bucket = cluster[0]
        collection = cluster[1]
        for info_dict in collection:
            for title, info in info_dict.items():
                for term_id, term_info in info['frame info'].items():
                    lemma = term_info['lemma']
                    predicates.append(lemma)

    vocabulary = list(set(predicates))
    return vocabulary

def compile_predicates(collections):
    """returns dictionary with {event type: {frame: [list of predicates_pos]}"""
    event_type_frame_predicate_dict = {}

    for event_type, info_list in collections.items():
        frame_predicate_dict = defaultdict(list)
        for nafdict in info_list:
            for title, info in nafdict.items():
                for term_id, frame_info in info['frame info'].items(): #iterate over term id:frame info
                    frame = frame_info['frame']
                    lemma = frame_info['lemma']
                    pos = frame_info['POS']
                    lemma_pos = f'{lemma}_{pos}'
                    frame_predicate_dict[frame].append(lemma_pos)
        event_type_frame_predicate_dict[event_type] = frame_predicate_dict
    return event_type_frame_predicate_dict

def frequency_distribution(event_type_frame_predicate_dict):
    """calculates frequency distribution of predicates for each frame of an event_type and returns dictionary"""
    frame_predicate_distribution_dict = {}

    for event_type, frame_dict in event_type_frame_predicate_dict.items():
        frequency_dict = {}
        for frame in frame_dict:
            lemma_frequency_dict = {}
            counter = Counter(frame_dict[frame]) #create variable with frequency distribution
            count = 0
            for key in counter: #iterate over variable
                count += counter[key] #add frequency of each lemma to count
            for lemma, value in counter.items(): #iterate over both the lemma's and frequencies
                occurence_frequency = {}
                relative_frequency = (value/count)*100 #create variable with relative frequency
                occurence_frequency['absolute frequency'] = value #add occurences to nested dictionary
                occurence_frequency['relative frequency'] = relative_frequency #add frequency to nested dictionary
                lemma_frequency_dict[lemma] = occurence_frequency #add nested dictionary to nested dictionary
            frequency_dict[frame] = lemma_frequency_dict #add nested dictionary to dictionary
        frame_predicate_distribution_dict[event_type] = frequency_dict
    return frame_predicate_distribution_dict

def create_output_folder(output_folder,start_from_scratch):
    '''creates output folder for export dataframe'''
    folder = output_folder

    if os.path.isdir(folder):
        if start_from_scratch == True:
            shutil.rmtree(folder)

    if not os.path.isdir(folder):
        os.mkdir(folder)

def output_predicates_to_format(frequency_dict,xlsx_path,output_folder,start_from_scratch):
    """exports the predicate distribution to an excel format"""
    headers = ['event type', 'frame', 'lemma_POS', 'absolute freq', 'relative freq']
    list_of_lists = []

    for event_type in frequency_dict:
        for frame in frequency_dict[event_type]:
            for lemma_pos in frequency_dict[event_type][frame]:
                one_row = [event_type, frame, lemma_pos]
                for key, value in frequency_dict[event_type][frame][lemma_pos].items():
                    one_row.append(value)
                list_of_lists.append(one_row)

    df = pd.DataFrame(list_of_lists, columns=headers)

    if output_folder != None:
        create_output_folder(output_folder=output_folder,
                            start_from_scratch=start_from_scratch)
        df.to_excel(xlsx_path, index=False)
    return df
