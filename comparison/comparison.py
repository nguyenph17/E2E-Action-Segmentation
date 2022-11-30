
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append(os.path.abspath(os.path.join('..', 'visualization')))
from visualization import PlotSegments

# Now do your import


def load_label(numpy_filepath):
    labels = np.load(numpy_filepath)
    return labels


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def compute_error(start_1, end_1, start_2, end_2):
    intersection = np.minimum(end_1, end_2) - np.maximum(start_1, start_2)
    if intersection < 0:
        intersection = 0
    union = np.maximum(end_1, end_2) - np.minimum(start_1, start_2)
    IoU = (1.0*intersection / union)
    return 1 - IoU


def levenstein(p, y, p_starts, p_ends, y_starts, y_ends, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                #error = compute_error(p_starts[i-1], p_ends[i-1], y_starts[j-1], y_ends[j-1])
                D[i, j] = D[i-1, j-1]  # + error
            else:
                error = 1
                D[i, j] = min(D[i-1, j] + error,
                              D[i, j-1] + error,
                              D[i-1, j-1] + error)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def process(pred_npy_path, ds_files, video_path):
    with open(ds_files) as file:
        groundtruth_paths = [line.rstrip() for line in file]

    label_1 = load_label(pred_npy_path)
    segments_1, starts_1, ends_1 = get_labels_start_end_time(label_1)

    scores = {}
    for file in groundtruth_paths:
        
        groundtruth_name = os.path.split(file)[-1]
        pred_filename = os.path.split(pred_npy_path)[-1]

        if groundtruth_name == pred_filename:
            continue

        label_2 = load_label(file)
        segments_2, starts_2, ends_2 = get_labels_start_end_time(label_2)
        score = levenstein(segments_1, segments_2, starts_1,
                           ends_1, starts_2, ends_2)
        score = 1 - (score / max(len(segments_1), len(segments_2)))
        print(score)
        scores[file] = score

    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    scores = list(scores.items())[:5]
    return scores   



if __name__ == '__main__':
    pred_npy_path = 'feature_extraction/outputs/features_30fps/gt_arr/rgb-03-2.npy'
    video_path = 'E:/AICamp/Human-Action-Reconigtion-Comparison/rgb/rgb/rgb-03-2.avi'

    dataset_file = 'comparison/label_file.txt'

    process(pred_npy_path, dataset_file, video_path=video_path)

