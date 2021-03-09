from collections import defaultdict
from .fficf_utils import frames_collection, create_output_folder
import operator
import random
import json

def get_historical_distance(collections,lang2doc2dct_info,language,unknown_distance_path,output_folder,start_from_scratch,verbose):
    """creates a nested dictionary with event type as key and a sorted list of tuples, sorted on historical distance as value"""
    event_type_historical_distance_dict = defaultdict(list)
    event_type_unknown_distace_dict = {}

    for event_type, info_dicts in collections.items():
        historical_distance_dict = defaultdict(list)
        unknown_distance = []
        count = 0
        for info_dict in info_dicts:
            for title, info in info_dict.items():
                for doc in lang2doc2dct_info[language]:
                    if doc == title:
                        historical_distance = lang2doc2dct_info[language][doc]['historical distance']
                        if historical_distance == "unknown":
                            if verbose >= 3:
                                count += 1
                            unknown_distance.append(info_dict)
                            continue
                        else:
                            assert type(historical_distance) == int, "historical distance is not integer"
                            historical_distance_dict[historical_distance].append(info_dict)
        event_type_unknown_distace_dict[event_type] = unknown_distance
        create_output_folder(output_folder=output_folder,
                            start_from_scratch=start_from_scratch,
                            verbose=verbose)
        with open(unknown_distance_path, 'w') as outfile:
            json.dump(event_type_unknown_distace_dict, outfile, indent=4, sort_keys=True)
        if verbose >= 3:
            print(f"{event_type}: {count} texts with unknown historical distance filtered out")
        sorted_days = sorted(historical_distance_dict.items(), key=operator.itemgetter(0))
        event_type_historical_distance_dict[event_type] = sorted_days
    return event_type_historical_distance_dict

def historical_distance_frames(historical_distance_dict):
    """extract frames per day and put them in a list"""
    historical_distance_frames = {}

    for event_type, list_of_tuples in historical_distance_dict.items():
        new_list_tuples = []
        for tupl in list_of_tuples:
            day = tupl[0]
            collection = tupl[1]
            frames = frames_collection(collection)
            new_tuple = (day, frames)
            new_list_tuples.append(new_tuple)
        historical_distance_frames[event_type] = new_list_tuples
    return historical_distance_frames

def cluster_time_buckets(historical_distance_dict,time_buckets,verbose):
    """clusters the info_dicts under provided time buckets"""
    time_bucket_dict = {}

    for event_type, sorted_days in historical_distance_dict.items(): #iterate over event type:list of tuples
        time_bucket_clusters = []
        for time_bucket, rang in time_buckets.items(): #iterate over time bucket:range
            info_dicts = []
            for tupl in sorted_days: #iterate over list of tuples
                day = tupl[0] #historical distance
                info = tupl[1] #list of information dictionaries
                if day in rang:
                    info_dicts.append(info) #append the list of information dictionaries to list
            flat_list = [item for sublist in info_dicts for item in sublist] #convert list of lists to one list with dictionaries
            time_bucket_tuple = (time_bucket, flat_list) #create tuple with time bucket and list with dictionaries
            time_bucket_clusters.append(time_bucket_tuple) #append tuple to list
        time_bucket_dict[event_type] = time_bucket_clusters #event type as key and list of time buckets with info as key

    if verbose >= 1:
        for event_type, time_buckets in time_bucket_dict.items():
            print(event_type,":")
            for cluster in time_buckets:
                print(f"time bucket {cluster[0]}: {len(cluster[1])} reference texts")

    return time_bucket_dict

def sample_time_buckets(historical_distance_info_dict, verbose):
    """create proportional sizes of subcorpora across time buckets. randomize when selecting the texts for this sample."""
    event_type_lengths_dict = {}

    for event_type, clusters in historical_distance_info_dict.items(): #iterate over clusters of dicts
        lengths_dict = {}
        for cluster in clusters:
            time_bucket = cluster[0]
            list_of_dicts = cluster[1]
            lengths_dict[time_bucket] = len(list_of_dicts) #add time bucket:number of docs
            event_type_lengths_dict[event_type] = lengths_dict

    sampled_collection_dict = {}

    for event_type, clusters in historical_distance_info_dict.items(): #iterate over clusters of dicts
        list_of_buckets = []
        lengths_dict = event_type_lengths_dict[event_type]
        len_smallest_corpus = min(lengths_dict.values())
        if verbose >= 1:
            print(f"{event_type}: {len_smallest_corpus} sampled texts per time bucket")
        for cluster in clusters:
            time_bucket = cluster[0]
            list_of_dicts = cluster[1]
            sampled_list = random.sample(list_of_dicts, len_smallest_corpus) #extract random sample with the length of the smallest subcorpus
            assert len(sampled_list) == len_smallest_corpus, "subcorpora not proportional"
            new_cluster = (time_bucket, sampled_list) #create new cluster with time bucket and samples subcorpus
            list_of_buckets.append(new_cluster) #append cluster to list
        assert len(list_of_buckets) == len(lengths_dict), "not all time buckets represented"
        sampled_collection_dict[event_type] = list_of_buckets #add event type:list of clusters to dict
    return sampled_collection_dict

def collection_to_json(sampled_collection,json_path,output_folder,start_from_scratch,verbose):
    """export collections to json"""
    create_output_folder(output_folder=output_folder,
                        start_from_scratch=start_from_scratch,
                        verbose=verbose)
    with open(json_path, 'w') as outfile:
        json.dump(sampled_collection, outfile, indent=4, sort_keys=True)
