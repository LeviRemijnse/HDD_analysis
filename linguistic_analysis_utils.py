import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_frequency(historical_distance_frames_dict,event_type,frames,pdf_path,verbose):
    """extract frequencies of specified frames per time bucket and visualize"""
    headers = ["time bucket", "frame", "frequency"]
    list_of_lists = []

    for frame in frames:
        for tupl in historical_distance_frames_dict[event_type]:
            one_row = []
            time_bucket = tupl[0]
            one_row.append(time_bucket)
            clustered_frames = tupl[1]
            frequencies = Counter(clustered_frames)
            one_row.append(frame)
            one_row.append(frequencies[frame])
            list_of_lists.append(one_row)

    df = pd.DataFrame(list_of_lists, columns=headers)
    lineplot = sns.lineplot(data=df, x="time bucket", y="frequency", hue="frame")
    plt.savefig(pdf_path)

    return
    
