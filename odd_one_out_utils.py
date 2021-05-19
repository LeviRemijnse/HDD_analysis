import pandas as pd
import numpy as np
import networkx as nx
import operator
from .frame_relation_utils import get_frame_to_root_information, get_digraph, get_relations, get_graph
from collections import defaultdict
from itertools import combinations
from functools import lru_cache

def get_frame_to_root_info(fn, verbose):
    """returns dictionary with per frame the root, path to the root and length of the path"""
    sub_to_super = get_relations(fn=fn, relations={'Inheritance'}, verbose=1)
    g, roots = get_digraph(sub_to_super, verbose=1)
    frame_to_root_info = get_frame_to_root_information(di_g=g, fn=fn, roots=roots, verbose=1)
    return frame_to_root_info

def get_root_to_frame_info(frame_to_root_info, verbose):
    """returns a dictionary with per root the frame, path to root and length of the path"""
    root_to_frame_dict = defaultdict(list)

    for frame, info_list in frame_to_root_info.items():
        for info_dict in info_list:
            root = info_dict['root']
            path_dict = {'the_path': info_dict['the_path'], 'len_path': info_dict['len_path']}
            frame_dict = {frame: path_dict}
            root_to_frame_dict[root].append(frame_dict)

    root_to_frame_info_dict = {}

    for root, framelist in root_to_frame_dict.items():
        frame_dict = {}
        for framedict in framelist:
            for frame, info in framedict.items():
                frame_dict[frame] = info
        root_to_frame_info_dict[root] = frame_dict

    return root_to_frame_info_dict

def extract_rel_freq(frames_path, frame_to_root_info, event_type, verbose):
    """extract frames and their relative frequencies from excel file and return a list of sorted tuples"""
    df = pd.read_excel(frames_path)
    df_event_type = df.loc[df['event type'] == event_type]

    root_rel_freqs = defaultdict(set)
    count = 0

    for index, row in df_event_type.iterrows():
        if row['relative freq'] != 0:
            frame = row['frame']
            count += 1
            rel_freq = row['relative freq']
            roots = {root_info['root'] for root_info in frame_to_root_info[frame]}
            tupl = (frame, rel_freq)
            for root in roots:
                root_rel_freqs[root].add(tupl)

    root_rel_freqs_sorted = {k: v for k, v in sorted(root_rel_freqs.items(), key=lambda item: len(item[1]), reverse=True)}

    if verbose >= 1:
        print(f"{count} frames with occurrences in {event_type}")
        print()
    if verbose >= 3:
        for root, frames in root_rel_freqs_sorted.items():
            print(f"{root}: {len(frames)} frames")

    return root_rel_freqs

def split_rel_freq(root_rel_freqs, range_dict, root, verbose):
    """takes a sorted set and splits it into three parts based on preset ratios"""
    split_dict = defaultdict(list)

    for tupl in root_rel_freqs[root]:
        frame = tupl[0]
        rel_freq = tupl[1]
        for cat, (minimum, maximum) in range_dict.items():
            if rel_freq >= minimum and rel_freq < maximum:
                split_dict[cat].append(frame)

    if verbose >= 3:
        print()
        print(root)
        for cat, lst in split_dict.items():
            print(f"{len(lst)} frames in {cat} after splitting")
    return split_dict

#@lru_cache()
def get_shortest_path(sampled_items, fn, verbose):
    """filter items based on path length between the frames"""
    short_path_items_dict = {}
    sub_to_super = get_relations(fn=fn, relations=set(), verbose=verbose)
    g = get_graph(sub_to_super=sub_to_super, verbose=verbose)
    for root, cats in sampled_items.items(): #enter the root
        short_path_cats_dict = {}
        for cat, items in cats.items(): #enter the relative frequency category
            short_path_items = []
            for item in items: #enter the items
                for frame1, frame2 in combinations(item, 2): #enter the item
                    path_frame1_frame2 = nx.shortest_path(g, frame1, frame2)
                    for frame in item:
                        if frame != frame1 and frame != frame2:
                            frame3 = frame
                    path_frame1_frame3 = nx.shortest_path(g, frame1, frame3)
                    path_frame2_frame3 = nx.shortest_path(g, frame2, frame3)
                    if all([len(path_frame1_frame2) <= 3,
                            len(path_frame1_frame3) >= 4,
                            len(path_frame2_frame3) >= 4]):
                        short_path_items.append(item)
            short_path_cats_dict[cat] = short_path_items
        short_path_items_dict[root] = short_path_cats_dict
    return short_path_items_dict

#@lru_cache()
def get_paths_to_root(sampled_items, root_to_frame_info, verbose):
    """filter items based on the distance to their root"""
    root_path_items_dict = {}
    for root, cats in sampled_items.items(): #enter the root
        root_path_cats_dict = {}
        for cat, items in cats.items(): #enter the relative frequency category
            root_path_items = set()
            for item in items: #enter the list of items
                for frame1, frame2 in combinations(item, 2): #enter the item
                    depth_to_root_frame1 = root_to_frame_info[root][frame1]['len_path']
                    depth_to_root_frame2 = root_to_frame_info[root][frame2]['len_path']
                    for frame in item: #TODO make helper function, return 3 permutations
                        if frame != frame1 and frame != frame2:
                            frame3 = frame
                    depth_to_root_frame3 = root_to_frame_info[root][frame3]['len_path']
                    if all([depth_to_root_frame1 <= 4,
                            depth_to_root_frame2 >= 5,
                            depth_to_root_frame3 >= 5]):
                        root_path_items.add((frame1, frame2, frame3))
            root_path_cats_dict[cat] = root_path_items
        root_path_items_dict[root] = root_path_cats_dict
    return root_path_items_dict

def sample_items(split_dict, item_length, use_shortest_path, fn, frame_to_root_info, root_to_frame_info, use_shortest_path_to_root, verbose):
    """sample items"""
    random_samples = {}

    for root, cats in split_dict.items():
        cat_samples = {}
        for cat, frames in cats.items():
            combs = list(combinations(frames, item_length))
            cat_samples[cat] = combs
        random_samples[root] = cat_samples

    if verbose:
        print()
        for root, cats in random_samples.items():
            print(root)
            for cat, items in cats.items():
                print(f"{cat.upper()}: {len(items)} items after sampling combinations")
                #for item in items:
                #    print(item)
                #print()

    if use_shortest_path == True:
        short_path_items = get_shortest_path(sampled_items=random_samples,
                                        fn=fn,
                                        verbose=verbose)

        if verbose:
            print()
            print("SHORTEST PATH SAMPLING")
            for root, cats in short_path_items.items():
                print(root)
                for cat, items in cats.items():
                    print(f"{cat.upper()}: {len(items)} items")
                    #for item in items:
                    #    print(item)

    if use_shortest_path_to_root == True:
        shortest_path_to_root_items = get_paths_to_root(sampled_items=random_samples,
                                                    root_to_frame_info=root_to_frame_info,
                                                    verbose=verbose)
        if use_shortest_path_to_root == True and verbose:
            print()
            print("SHORTEST PATH TO ROOT SAMPLING")
            for root, cats in shortest_path_to_root_items.items():
                print(root)
                for cat, items in cats.items():
                    print(f"{cat.upper()}: {len(items)} items")
                for item in cats["low"]:
                    print(item)
    return random_samples
