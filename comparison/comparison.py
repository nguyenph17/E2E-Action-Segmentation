import numpy as np
import sys, os
#sys.path.append(os.path.abspath(os.path.join('..', 'visualization')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now do your import
from visualization import PlotSegments


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
                D[i, j] = D[i-1, j-1] #+ error
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





if __name__ == '__main__':
    gt_arr_1 = 'feature_extraction/outputs/features_30fps/gt_arr/rgb-03-2.npy'
    video_path = 'E:/AICamp/Human-Action-Reconigtion-Comparison/rgb/rgb/rgb-03-2.avi'


    dataset_file = 'comparison/label_file.txt'
    with open(dataset_file) as file:
        groundtruth_paths = [line.rstrip() for line in file]

    label_1 = load_label(gt_arr_1)
    segments_1, starts_1, ends_1 = get_labels_start_end_time(label_1)

    best_score = 0.0
    similar_label = ''
    for file in groundtruth_paths:
        label_2 = load_label(file)
        segments_2, starts_2, ends_2 = get_labels_start_end_time(label_2)
        score = levenstein(segments_1, segments_2, starts_1, ends_1, starts_2, ends_2) 
        score = 1 - (score / max(len(segments_1), len(segments_2)))
        print(score)
        if best_score < score:
            best_score = score
            similar_label = file

    print(f"The most similar file is: {similar_label} \n The score is: {best_score}")
    plot_segments = PlotSegments('./visualization/mapping.txt')
    plot_segments.visualize(video_path ,gt_arr_1, gt_arr_1)