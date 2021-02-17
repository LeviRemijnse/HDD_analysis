from .DFNDataReleases import get_relevant_info
from .DFNDataReleases import dir_path as REPO_DIR
from collections import defaultdict
import os
import json

def get_naf_paths(project,language,verbose=0):
    """
    Get a dictionary with event type as key and a set of NAF paths as value.
    :param project: the project under which the NAF files are generated.
    :param language: the language of the reference texts.
    :type project: string
    :type language: string
    """
    relevant_info = get_relevant_info(repo_dir=REPO_DIR,
                                    project=project,
                                    load_jsons=True)
    event_type_collection = defaultdict(set)

    incidents = relevant_info['proj2inc'][project]
    for incident in incidents:
        event_type = relevant_info['inc2type'][incident]
        doc_list = relevant_info['inc2lang2doc'][incident][language]
        for doc in doc_list:
            path = os.path.join(relevant_info["unstructured"], language, f"{doc}.naf")
            assert os.path.exists(path), f"{path} does not exist on disk"
            event_type_collection[event_type].add(path)
    if verbose >= 2:
        for event_type, collection in event_type_collection.items():
            print(f'{event_type}: {len(collection)} reference texts')
    return event_type_collection

def get_lang2doc2dct_info(project,language, verbose):
    """
    get the path to the json with historical distance in days per document to the event
    """
    relevant_info = get_relevant_info(repo_dir=REPO_DIR,
                                    project=project,
                                    load_jsons=True)
    path = os.path.join(relevant_info['project_statistics'], 'lang2doc2dct_info.json')
    assert os.path.exists(path), f"{path} does not exist on disk"

    with open(path, "r") as infile:
        historical_distance_dict = json.load(infile)

    if verbose >= 1:
        print(f"lang2doc2dct_info contains historical distance for {len(historical_distance_dict[language])} documents")
    return historical_distance_dict
