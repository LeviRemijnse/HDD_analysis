from nltk.corpus import framenet as fn
from collections import defaultdict, Counter
import networkx as nx

def get_relations(fn, relations=set(), verbose=0):
    """get the inheritance relations for the FrameNet frames"""
    frame_to_superframe = defaultdict(set)

    for fn_rel in fn.frame_relations():
        if relations:
            if fn_rel.type.name not in relations:
                continue
        frame_to_superframe[fn_rel.subFrameName].add(fn_rel.superFrameName)

    if verbose:
        print()
        print(f'found {len(frame_to_superframe)} frames with at least one superframe.')
        print(f'chosen relations are {relations} (empty set indicates all frame-to-frame relations).')
        print(Counter(len(value) for value in frame_to_superframe.values()))
    return frame_to_superframe

def get_digraph(sub_to_super, verbose=0):
    """get digraph"""
    g = nx.DiGraph()

    for subframe, superframes in sub_to_super.items():
        for superframe in superframes:
            g.add_edge(superframe, subframe)

    roots = set()

    for node in g.nodes():
        parents = list(g.predecessors(node))
        if not parents:
            roots.add(node)

    if verbose >= 1:
        print()
        print(nx.info(g))
        print(f'number of roots is {len(roots)}')
    return g, roots

def get_graph(sub_to_super, verbose=0):

    g = nx.Graph()
    for subframe, superframes in sub_to_super.items():
        for superframe in superframes:
            g.add_edge(superframe, subframe)

    if verbose >= 1:
        print()
        print(nx.info(g))
    return g

def get_frame_to_root_information(di_g, fn, roots, verbose=0):
    """get all the relations from frames to their roots"""
    frame_to_root_information = {}

    for frame_obj in fn.frames():
        frame = frame_obj.name
        if not di_g.has_node(frame):
            root_information = [{
                'subframe' : frame,
                'root' : frame,
                'the_path' : [frame],
                'len_path' : 1
            }]
        else:
            root_information = []
            for root in roots:
                if nx.has_path(di_g, root, frame):
                    the_path = nx.shortest_path(di_g, root, frame)
                    len_path = len(the_path)
                    root_info = {
                        'subframe' : frame,
                        'root' : root,
                        'the_path' : the_path,
                        'len_path' : len_path
                    }
                    root_information.append(root_info)
        # check for 2> root paths
        #chosen_root_info = {}
        #min_path_length = 100000

        #for root_info in root_information:
        #    if root_info['len_path'] < min_path_length:
        #        min_path_length = root_info['len_path']
        #        chosen_root_info = root_info
        #assert chosen_root_info != {}

        frame_to_root_information[frame] = root_information

    assert len(frame_to_root_information) == 1221
    #path_lengths = [root_info['len_path']
    #                for root_info in frame_to_root_information.values()]

    #if verbose >= 1:
    #    print()
    #    print(f'distribution of path lengths: {Counter(path_lengths)}')
    return frame_to_root_information
