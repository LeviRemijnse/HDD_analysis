### Historical Distance Data Analysis
This package provides functions that extract linguistic information from a corpus of reference texts categorized into event types and ranging in historical distance from days to years after the event. With this linguistic information, one can perform two kinds of analyses. First, one can train a diagnostic classifier with different features extracted from the gun violence corpus. Second, one can apply an FF*ICF metric on the frames between the event types in order to derive the typical frames for the event type.

This package was built and used for the purpose of the paper Variation in framing as a function of temporal reporting distance; Remijnse et al. 2021.

### Prerequisites
Python 3.7.4 was used to create this package. It might work with older versions of Python.

### Installing

# Resources
A number of GitHub repositories need to be cloned. This can be done calling:
```bash
bash install.sh
```

# Python modules
A number of external modules need to be installed, which are listed in **requirements.txt**.
Depending on how you installed Python, you can probably install the requirements using one of following commands:
```bash
pip install -r requirements.txt
```

### Usage
This package comes with different main functions:

# Frame information from a file
The function frame_info() extracts the following linguistic information from a NAF iterable. You can run the function on an example file with the following command:

```python
from HDD_analysis import frame_info, dir_path

frame_info(naf_root=f"{dir_path}/test/input_files/VJ Cozer - Creator Entertainment.naf",
            verbose=0)
```
The following parameters are specified:
* **naf_root** the path to a NAF file
* **verbose**: 2: print the extracted information
When running this function in python, the output will be printed in your terminal.

# Classifier training
The function linguistic_analysis() is used to crawl gun violence reference texts, extract both the annotated frames and their predicates, train a Support Vector and evaluate its performance. You can run the following code in your terminal.

```python
from HDD_analysis import linguistic_analysis, dir_path

time_buckets = {"day_0":range(0,1), "day_8-30":range(7,31)}

linguistic_analysis(time_bucket_config=time_buckets,
                        absolute_path=f"{dir_path}/test/output",
                        experiment="frequency",
                        project='GVA',
                        language='en',
                        event_type="Q5618454",
                        path_typicality_scores=None,
                        use_frames=True,
                        discourse_sensitive=False,
                        use_bow=False,
                        balanced_classes=True,
                        verbose=5)
```
The following parameters are specified:
* **time_bucket_config** the configuration of time buckets
* **absolute_path** output folder
* **experiment** here you type the variables of the experiment, in this case "frequency" for the frame frequency
* **project** name of the project in DFNDataReleases
* **language** the language of reference texts in the project
* **event_type** Wikidata identifier of the event type
* **path_typicality_scores** the path to json with typicality scores
* **use_frames** indicate whether you want to use frame frequency
* **discourse_sensitive** indicate whether you want to use discourse ratio of the frames
* **use_bow** indicate whether you want to use predicate frequencies
* **balanced_classes** indicate whether you want to balance the corpora across time buckets
* **verbose** 1: print evaluation report, print base folder, print number of documents with historical distance, print number of documents removed, print number of documents per time bucket, if balanced classes: print number of documents per sampled time bucket, 2: print number of documents per event type, 3: print number of documents with unknown historical distance filtered out, 5: print length of dataframes, print number of frames with 0 occurrences across all rows and all dataframes, print number of columns per dataframe

After calling this function, the following folder structure is created in the output folder:
* timebucket1---timebucketn+(un)balanced
  * experiment
    error_analysis.xlsx
    model.pkl
    test_features.pkl
    test_report.txt
    train_features.pkl
  sampled_corpus.json
  titles_dev.pkl
  titles_test.pkl
  titles_train.pkl
  unknown_distance.json

# Contrastive analysis
fficf_info() crawls the NAF files from a specific project of DFNDataReleases, performs a contrastive analysis and writes the output to different formats. You can run the function with the code below. This code integrates output data from linguistic analysis(), so make sure you run that function first.

```python
from HDD_analysis import fficf_info, dir_path

fficf_info(project='HDD',
            language='en',
            analysis_types=["c_tf_idf"],
            xlsx_paths=[f'{dir_path}/test/output/ff_icf.xlsx'],
            output_folder=f'{dir_path}/test/output',
            start_from_scratch=False,
            json_paths=[f'{dir_path}/test/output/ff_icf.json'],
            gva_path=f'{dir_path}/test/output/day_0---day_8-30+balanced/unknown_distance.json',
            verbose=3)
```
The following parameters are specified:
* **project** name of the project in DFNDataReleases
* **language** the language of the reference texts in the project
* **analysis_types** a list with the types of contrastive analyses you want the function to perform. The list now only contains "c_tf_idf" which is used in the paper. It contrasts all the frames per event type. It is also possible to append "tf_idf". It contrasts frames per reference texts and means over event types.
* **xlsx_paths** list of excel paths where the output is written to. The order of contents should be consistent with the content of analysis_types.
* **output_folder** output folder
* **start_from_scratch** boolean that indicates whether previous output should be overwritten
* **json_paths** list of json paths where the output is written to. The order of contents should be consistent with the content of analysis_types and xlsx_paths
* **gva_path** path to json with frame information for the event type gun violence, which is integrated for the paper. If this parameter is not specified, the function will only contrast event types of the project.
* **verbose** 1: print number of frames removed, 2: print number of reference texts per event type, number of frames per event type 3: print number of sampled reference texts per event type, show bottom 5 and top 5 ranking of frames per event type.

When running this function, the output of the contrastive analysis is written to 1) an excel file with a ranking of the annotated frames per event type, based on their FF*ICF scores. Frequency distributions are provided as well. 2) a json file with a dictionary containing {event type: {frame: score}}. This can be used to update DFNDataReleases.

### Authors
* **Levi Remijnse** (l.remijnse@vu.nl)

### License
This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details
