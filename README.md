### Historical Distance Data Analysis
This package provides functions that extract linguistic information from a corpus of reference texts categorized into event types and ranging in historical days from days to years after the event. With this linguistic information, one can perform two kinds of analyses. First, one can apply an FF*ICF metric on the frames between the event types in order to derive the typical frames for the event type. Second, one can perform different linguistic analyses of the corpus of an event type by taking measuring the distribution of the linguistic information over the historical distance.

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

# Contrastive analysis
fficf_info() crawls the NAF files for a specific project from DFNDataReleases, and writes an output file with a ranking of the annotated frames per event type, based on their FF*ICF scores. Frequency distributions are provided as well.

# Get paths to NAF files
The function get_paths() takes the name of the research project and the language and returns a dictionary with event type as key and a set of paths to the NAF files as value.

# Frame information from a file
The function frame_info() extracts the following linguistic information from a NAF iterable:
- the total number of frames as an integer
- for each frame the following information:
  - the frame
  - the predicate
  - POS
  - when the predicate is part of a compound: compound information
  - when the predicate is modified by an article: article information

# Frame information for an event type
event_type_info() iterates over a collection of collections of NAF paths per event type, and calls frame_info() on each file. It returns a dictionary with event type as key and a list of dictionaries with linguistic information.

##Notes


## Authors
* **Levi Remijnse** (l.remijnse@vu.nl)
