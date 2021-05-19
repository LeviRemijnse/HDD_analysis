from .xml_utils import srl_id_frames, term_id_lemmas, determiner_id_info, compound_id_info, get_text_title, frame_info_dict, sentence_info
from .fficf_utils import contrastive_analysis, output_tfidf_to_format, sample_corpus, delete_smallest_texts, output_tfidf_to_json, create_output_folder, compute_c_tf_idf_between_time_buckets
from .predicate_utils import compile_predicates, frequency_distribution, output_predicates_to_format, output_predicates_to_json
from .path_utils import get_naf_paths, get_lang2doc2dct_info, analysis_paths
from .historical_distance_utils import get_historical_distance, historical_distance_frames, cluster_time_buckets, sample_time_buckets, collection_to_json
from .linguistic_analysis_utils import doc_features, split_df, extract_features_and_labels, train_classifier, classify_data, evaluation, titles_buckets_df, df_to_pickle, get_frame_vocabulary, bag_of_predicates, remove_columns_with_zeros
from .error_analysis_utils import f_importances
from .odd_one_out_utils import extract_rel_freq, get_frame_to_root_info, split_rel_freq, sample_items, get_root_to_frame_info

from lxml import etree
from nltk.corpus import framenet as fn
import json
import os
import pickle
import pandas as pd

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
                json_paths=None,
                gva_path=None,
                verbose=0):
    """
    Extract frames dictionary with naf information per text per event type, perform fficf metrics and returns a dataframe.
    :param project: name of the project in DFNDataReleases
    :param language: language of the reference texts in the project
    :param analysis_types: list with the types of contrastive analyses (c_tf_idf or tf_idf)
    :param xlsx_paths: list of excel paths where the output is written to
    :param output_folder: output folder
    :param start_from_scratch: should previous output should be overwritten?
    :param stopframes: frames that should be ignored when performing tf_idf
    :param min_df: when vectorizing, ignore terms that have a document frequency lower than this threshold
    :param minimal_frames_per_doc: ignore documents with frame frequency lower than this threshold
    :param json_paths: list of json paths where the output is written to
    :param gva_path: path to gun violence subcorpus
    :type project: string
    :type language: string
    :type analysis_types: list
    :type xlsx_paths: list
    :type output_folder: string
    :type start_from_scratch: boolean
    :type stopframes: list
    :type mind_df: integer
    :type minimal_frames_per_doc: integer
    :type json_paths: list
    :type gva_path: string
    """
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
    if gva_path != None:
        with open(gva_path, "r") as infile:
            gva_dict = json.load(infile)
        for event_type, info_dicts in gva_dict.items():
            sliced_corpus[event_type] = info_dicts

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
                                start_from_scratch=start_from_scratch,
                                verbose=verbose)

    if json_paths != None:
        for output_dict, json_path in zip(output_list, json_paths):
            output_tfidf_to_json(tf_idfdict=output_dict,
                                    json_path=json_path,
                                    output_folder=output_folder,
                                    start_from_scratch=start_from_scratch,
                                    verbose=verbose)
    return

def frame_predicate_distribution(collections,
                                    xlsx_path,
                                    output_folder,
                                    json_path,
                                    start_from_scratch=False,
                                    verbose=0):
    """extracts a dictionary with event type as key and frame/predicate frequency distributions as values and returns a dataframe"""
    frames_predicates = compile_predicates(collections)
    distribution_dict = frequency_distribution(frames_predicates)
    distribution_dataframe = output_predicates_to_format(frequency_dict=distribution_dict,
                                                            xlsx_path=xlsx_path,
                                                            output_folder=output_folder,
                                                            start_from_scratch=start_from_scratch)
    output_predicates_to_json(frequency_dict=distribution_dict,
                                json_path=json_path,
                                output_folder=output_folder,
                                start_from_scratch=start_from_scratch,
                                verbose=verbose)
    return distribution_dataframe

def historical_distance(project,
                        language,
                        output_folder,
                        json_path,
                        unknown_distance_path,
                        balanced_classes,
                        start_from_scratch=False,
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
                                                        unknown_distance_path=unknown_distance_path,
                                                        output_folder=output_folder,
                                                        start_from_scratch=start_from_scratch,
                                                        verbose=verbose)

    if verbose >= 1:
        for event_type, sorted_days in historical_distance_dict.items():
            close = min(sorted_days)
            far = max(sorted_days)
            print(f"{event_type}: {len(close[1])} texts published {close[0]} days after the event, {len(far[1])} texts published {far[0]} days after the event")

    if dct_time_buckets != None:
        time_bucket_dict = cluster_time_buckets(historical_distance_dict=historical_distance_dict,
                                                    time_buckets=dct_time_buckets,
                                                    verbose=verbose)
        if balanced_classes:
            sampled_corpus = sample_time_buckets(historical_distance_info_dict=time_bucket_dict,
                                                verbose=verbose)
        else:
            sampled_corpus = time_bucket_dict
        collection_to_json(sampled_collection=sampled_corpus,
                            json_path=json_path,
                            output_folder=output_folder,
                            start_from_scratch=start_from_scratch,
                            verbose=verbose)
        return sampled_corpus

    sampled_corpus = sample_time_buckets(historical_distance_info_dict=historical_distance_dict,
                                            verbose=verbose)
    collection_to_json(sampled_collections=sampled_corpus,
                        json_path=json_path,
                        output_folder=output_folder,
                        start_from_scratch=start_from_scratch,
                        verbose=verbose)
    return sampled_corpus

def sampled_titles_to_dfs(historical_distance_info_dict,
                        event_type,
                        train_path,
                        dev_path,
                        test_path,
                        output_folder,
                        start_from_scratch=False,
                        verbose=0):
    """loads a sampled corpus for a specified event type, splits data into train, develop and test dataframes"""
    with open(historical_distance_info_dict, "r") as infile:
        historical_distance_info_dict = json.load(infile)

    list_of_buckets = historical_distance_info_dict[event_type]

    df = titles_buckets_df(list_of_buckets=list_of_buckets)
    train_df, dev_df, test_df = split_df(df=df)

    df_to_pickle(train_df=train_df,
                    dev_df=dev_df,
                    test_df=test_df,
                    train_path=train_path,
                    dev_path=dev_path,
                    test_path=test_path,
                    output_folder=output_folder,
                    start_from_scratch=start_from_scratch,
                    verbose=verbose)
    return train_df, dev_df, test_df

def linguistic_analysis(time_bucket_config,
                        absolute_path,
                        experiment,
                        project,
                        language,
                        event_type,
                        path_typicality_scores,
                        use_frames=True,
                        discourse_sensitive=True,
                        use_typicality_scores=False,
                        use_bow=False,
                        balanced_classes=False,
                        verbose=0):
    """performs statistical analyses on linguistic phenomena distributed over time buckets and visualizes them"""
    paths_dict = analysis_paths(time_bucket_config=time_bucket_config, #create dictionary with paths
                                absolute_path=absolute_path,
                                experiment=experiment,
                                balanced_classes=balanced_classes,
                                verbose=verbose)

    if not os.path.isdir(paths_dict["time bucket folder"]):
        sampled_corpus = historical_distance(project=project, #create sampled_corpus, write to disk
                                            language=language,
                                            output_folder=paths_dict["time bucket folder"],
                                            json_path=paths_dict["sampled corpus"],
                                            unknown_distance_path=paths_dict["unknown distance"],
                                            balanced_classes=balanced_classes,
                                            dct_time_buckets=time_bucket_config,
                                            verbose=verbose)
        titles_train, titles_dev, titles_test = sampled_titles_to_dfs(historical_distance_info_dict=paths_dict["sampled corpus"], #create train, dev and test sets of document titles, write to disk
                                                                        event_type=event_type,
                                                                        train_path=paths_dict["train path"],
                                                                        dev_path=paths_dict["dev path"],
                                                                        test_path=paths_dict["test path"],
                                                                        output_folder=paths_dict["time bucket folder"],
                                                                        verbose=verbose)

    else:
        with open(paths_dict["sampled corpus"], "r") as infile:
            sampled_corpus = json.load(infile) #open sampled corpus as dictionary
        titles_train = pd.read_pickle(paths_dict["train path"]) #open titles of train set
        titles_dev = pd.read_pickle(paths_dict["dev path"]) #open titles of dev set
        titles_test = pd.read_pickle(paths_dict["test path"]) #open titles of test set

    create_output_folder(paths_dict['experiment folder'], start_from_scratch=True, verbose=verbose) #create subfolder for experiment

    vocabulary = get_frame_vocabulary(historical_distance_dict=sampled_corpus, #create frame vocabulary
                                        event_type=event_type)
    basekey_and_df = [("train", titles_train),("dev", titles_dev),("test", titles_test)] #list of tuples (basekey,split_titles_df)

    info = {}

    for basekey, split_df in basekey_and_df:
        if use_bow and use_frames:
            df_pred = bag_of_predicates(historical_distance_info_dict=sampled_corpus, #create dataframe with predicates as features
                                        event_type=event_type,
                                        titles_df=split_df,
                                        verbose=verbose)
            df_frame = doc_features(historical_distance_info_dict=sampled_corpus, #create dataframe with frames as features
                                    vocabulary=vocabulary,
                                    titles_df=split_df,
                                    event_type=event_type,
                                    discourse_sensitive=discourse_sensitive,
                                    path_typicality_scores=path_typicality_scores,
                                    use_typicality_scores=use_typicality_scores,
                                    verbose=verbose)
            del df_frame["time bucket"] #delete label column from DataFrame
            df = pd.concat([df_frame, df_pred], axis=1) #merge frame dataframe and predicate dataframe
        elif use_bow:
            df = bag_of_predicates(historical_distance_info_dict=sampled_corpus, #create dataframe with predicates as features
                                    event_type=event_type,
                                    titles_df=split_df,
                                    verbose=verbose)
        elif use_frames:
            df = doc_features(historical_distance_info_dict=sampled_corpus, #create dataframe with frames as features
                                vocabulary=vocabulary,
                                titles_df=split_df,
                                event_type=event_type,
                                discourse_sensitive=discourse_sensitive,
                                path_typicality_scores=path_typicality_scores,
                                use_typicality_scores=use_typicality_scores,
                                verbose=verbose)
        else:
            raise Exception(f'use_bow and use_frames are both False, which is not allowed')

        features_df_path  = os.path.join(paths_dict["experiment folder"], f'{basekey}_features.pkl')
        pickle.dump(df, open(features_df_path, 'wb'))
        info[f'{basekey}_df'] = df

    train_df, dev_df, test_df = remove_columns_with_zeros(train_df=info['train_df'], #remove all columns with zero's in dataframes
                                                            dev_df=info['dev_df'],
                                                            test_df=info['test_df'],
                                                            verbose=verbose)
    info['train_df'] = train_df #add training dataframe to info
    info['dev_df'] = dev_df #add dev dataframe to info
    info['test_df'] = test_df #add test dataframe to info

    if all([use_frames,
            not use_bow,
            path_typicality_scores]):
        with open(path_typicality_scores, "r") as infile:
            typicality_scores_dict = json.load(infile) #load typicality scores
        event_type_typicality_scores = typicality_scores_dict[event_type] #list of lists to variable
        for cutoff_point in [10, 25,50, 'all']:
            c_tf_idf_df = compute_c_tf_idf_between_time_buckets(typicality_scores=event_type_typicality_scores, #compute c-tf-idf between time buckets
                                                                train_df=train_df,
                                                                dev_df=dev_df,
                                                                test_df=test_df,
                                                                top_n_typical_frames=cutoff_point,
                                                                verbose=verbose)
            c_tf_idf_path = os.path.join(paths_dict['experiment folder'],f'{cutoff_point}.xlsx') #create path for excel
            c_tf_idf_df.to_excel(c_tf_idf_path) #export c-tf-idf between time buckets to excel


    for phase in ['train', 'dev', 'test']:
        features, labels = extract_features_and_labels(df=info[f'{phase}_df']) #extract features and labels
        info[f'{phase}_features'] = features #add features to info
        info[f'{phase}_labels'] = labels #add labels to info

    model, vec = train_classifier(train_features=info['train_features'], #create model and vectorizer of classifier
                                    train_targets=info['train_labels'])
    error_analysis_df = f_importances(model,info['train_df']) #model analysis to variable
    error_analysis_df.to_excel(paths_dict['error analysis path']) #model analysis to excel
    pickle.dump(model, open(paths_dict['model'], 'wb'))
    for phase in ['test']:
        predictions = classify_data(model=model, #run the model
                                    vec=vec,
                                    features=info[f'{phase}_features'])
        evaluation_report = evaluation(human_annotation=info[f'{phase}_labels'], #evaluate the model, evaluation to txt
                                        system_output=predictions,
                                        report_path=paths_dict[f'{phase} report'])

        if verbose >= 1:
            print(experiment, phase)
            print(evaluation_report)
    return

def prepare_odd_one_out(frames_path,
                        event_type,
                        range_dict,
                        roots,
                        item_length,
                        use_shortest_path=False,
                        use_shortest_path_to_root=False,
                        verbose=0):
    """open a frequency distribuation of frames for a specific event type and prepare sets of frames for odd
    one out annotation test."""
    frame_to_root_info_dict = get_frame_to_root_info(fn=fn,
                                                    verbose=verbose)
    root_to_frame_info_dict = get_root_to_frame_info(frame_to_root_info=frame_to_root_info_dict,
                                                    verbose=verbose)
    root_rel_freqs = extract_rel_freq(frames_path=frames_path,
                                        frame_to_root_info=frame_to_root_info_dict,
                                        event_type=event_type,
                                        verbose=verbose)
    split_dict = {}
    for root in roots:
        root_dict = split_rel_freq(root_rel_freqs=root_rel_freqs,
                                    range_dict=range_dict,
                                    root=root,
                                    verbose=verbose)
        split_dict[root] = root_dict
    sampled_items = sample_items(split_dict=split_dict,
                                    item_length=item_length,
                                    use_shortest_path=use_shortest_path,
                                    fn=fn,
                                    frame_to_root_info=frame_to_root_info_dict,
                                    root_to_frame_info=root_to_frame_info_dict,
                                    use_shortest_path_to_root=use_shortest_path_to_root,
                                    verbose=verbose)
