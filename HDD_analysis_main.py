from .xml_utils import srl_id_frames, term_id_lemmas, determiner_id_info, compound_id_info, get_text_title, frame_info_dict, sentence_info
from .fficf_utils import contrastive_analysis, output_tfidf_to_format, sample_corpus, delete_smallest_texts
from .predicate_utils import compile_predicates, frequency_distribution, output_predicates_to_format
from .path_utils import get_naf_paths, get_lang2doc2dct_info
from .historical_distance_utils import get_historical_distance, historical_distance_frames, cluster_time_buckets
from .linguistic_analysis_utils import visualize_frequency
from lxml import etree
import os

def frame_info(naf_root,
                verbose=0):
    '''
    extracts dictionary from naf with relevant info about frames and their lexical units.
    :param naf_iterable: path to naf iterable
    :type naf_iterable: string
    '''
    doc_tree = etree.parse(naf_root)
    root = doc_tree.getroot()

    title = get_text_title(root)
    framedict = srl_id_frames(root)
    lemmadict = term_id_lemmas(root)
    sentencedict = sentence_info(root)
    detdict = determiner_id_info(root)
    compounddict = compound_id_info(root)

    frame_info = frame_info_dict(title=title,
                                    framedict=framedict,
                                    lemmadict=lemmadict,
                                    sentencedict=sentencedict,
                                    detdict=detdict,
                                    compounddict=compounddict)
    if verbose >= 2:
        print(frame_info)
    return frame_info

def event_type_info(collections,
                verbose=0):
    """
    Returns a dictionary with event type as key and list of dictionaries with linguistic information as value.
    :param collections: a collection of collections of NAF paths per event type
    :type collections: dictionary
    """
    event_type_frame_info_dict = {}

    for event_type, collection in collections.items():
        collection_of_dicts = []
        for file in collection:
            frame_info_dict = frame_info(file)
            collection_of_dicts.append(frame_info_dict)
        event_type_frame_info_dict[event_type] = collection_of_dicts
    return event_type_frame_info_dict

def fficf_info(project,
                language,
                analysis_types,
                xlsx_paths,
                output_folder,
                start_from_scratch,
                stopframes=None,
                min_df=2,
                minimal_frames_per_doc=10,
                verbose=0):
    """extract frames dictionary with naf information per text per event type, perform fficf metrics and returns a dataframe"""
    assert type(analysis_types) == list, "analysis types are not in list"
    assert "tf_idf" or "c_tf_idf" in analysis_types, "metrics not recognized by fficf_info()"
    assert type(xlsx_paths) == list, "xlsx_paths not in list"

    event_type_paths_dict = get_naf_paths(project=project,
                                        language=language,
                                        verbose=verbose)
    event_type_info_dict = event_type_info(collections=event_type_paths_dict)
    sliced_corpus = delete_smallest_texts(collections=event_type_info_dict,
                                            minimal_n_frames=minimal_frames_per_doc,
                                            verbose=verbose)
    sampled_corpus = sample_corpus(sliced_corpus)

    if verbose >= 3:
        for event_type, info in sampled_corpus.items():
            print(f"{event_type}: {len(info)} sampled reference texts")

    output_list, frame_freq_dict = contrastive_analysis(collections=sampled_corpus,
                                                        analysis_types=analysis_types,
                                                        stopframes=stopframes,
                                                        min_df=min_df,
                                                        verbose=verbose)

    for output_dict, xlsx_path in zip(output_list, xlsx_paths):
        output_tfidf_to_format(tf_idfdict=output_dict,
                                frame_freq_dict=frame_freq_dict,
                                xlsx_path=xlsx_path,
                                output_folder=output_folder,
                                start_from_scratch=start_from_scratch)
    return

def frame_predicate_distribution(collections,
                                    xlsx_path,
                                    output_folder,
                                    start_from_scratch=False,
                                    verbose=0):
    """extracts a dictionary with event type as key and frame/predicate frequency distributions as values and returns a dataframe"""
    frames_predicates = compile_predicates(collections)
    distribution_dict = frequency_distribution(frames_predicates)
    distribution_dataframe = output_predicates_to_format(frequency_dict=distribution_dict,
                                                            xlsx_path=xlsx_path,
                                                            output_folder=output_folder,
                                                            start_from_scratch=start_from_scratch)
    return distribution_dataframe

def historical_distance(project,
                        language,
                        minimal_frames_per_doc=10,
                        dct_time_buckets=None,
                        verbose=0):
    """
    This function takes the project, language and time buckets and distributes the frames and linguistic information
    over the time buckets. It returns two dictionaries: 1) per event type a sorted list of tuples with
    (time_tbucket, [list of informative dictionaries per text]); 2) per event a sorted list of tuples with
    (time_bucket, [list of frames]). If no time buckets are provided, then the default setting is days.
    """
    lang2doc2dct_info = get_lang2doc2dct_info(project=project,
                                                language=language,
                                                verbose=verbose)
    event_type_paths_dict = get_naf_paths(project=project,
                                        language=language,
                                        verbose=verbose)
    event_type_info_dict = event_type_info(collections=event_type_paths_dict)
    sliced_corpus = delete_smallest_texts(collections=event_type_info_dict,
                                            minimal_n_frames=minimal_frames_per_doc,
                                            verbose=verbose)

    historical_distance_dict = get_historical_distance(collections=sliced_corpus,
                                                        lang2doc2dct_info=lang2doc2dct_info,
                                                        language=language,
                                                        verbose=verbose)
    historical_distance_frames_dict = historical_distance_frames(historical_distance_dict)

    if verbose >= 1:
        for event_type, sorted_days in historical_distance_dict.items():
            close = min(sorted_days)
            far = max(sorted_days)
            print(f"{event_type}: {len(close[1])} texts published {close[0]} days after the event, {len(far[1])} texts published {far[0]} days after the event")

    if dct_time_buckets != None:
        time_bucket_dict = cluster_time_buckets(historical_distance_dict=historical_distance_dict,
                                                    time_buckets=dct_time_buckets,
                                                    verbose=verbose)
        time_bucket_frames_dict = historical_distance_frames(time_bucket_dict)
        return time_bucket_dict, time_bucket_frames_dict

    return historical_distance_dict, historical_distance_frames_dict

def linguistic_analysis(historical_distance_frames_dict,
                        historical_distance_info_dict,
                        event_type,
                        frames,
                        pdf_path,
                        verbose=0):
    """performs statistical analyses on linguistic phenomena distributed over time buckets and visualizes them"""
    frequency = visualize_frequency(historical_distance_frames_dict,event_type,frames,pdf_path,verbose)

    return
