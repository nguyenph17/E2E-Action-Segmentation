import numpy as np
import pandas as pd
import os

def load_groundtruth(gt_path):
    groundtruth = np.load(gt_path)
    return groundtruth


def load_feats(feat_path):
    features = np.load(feat_path)
    return features


def get_segments(groundtruth, bg_class=["background"]):
        labels = []
        starts = []
        ends = []
        last_label = int(groundtruth[0])
        if groundtruth[0] not in bg_class:
            labels.append(int(groundtruth[0]))
            starts.append(0)
        for i in range(len(groundtruth)):
            if groundtruth[i] != last_label:
                if groundtruth[i] not in bg_class:
                    labels.append(int(groundtruth[i]))
                    starts.append(i)
                if last_label not in bg_class:
                    ends.append(i)
                last_label = groundtruth[i]
        if last_label not in bg_class:
            ends.append(i)
        return labels, starts, ends


def swap_segments(labels, starts, ends, i, j):
    labels[i], labels[j] = labels[j], labels[i]
    starts[i], starts[j] = starts[j], starts[i]
    ends[i], ends[j] = ends[j], ends[i]
    return labels, starts, ends




if __name__ == "__main__":
    video_file = 'augmentation/videos_to_swap.csv'
    groundtruth_dir = 'E:/AICamp/E2E-Action-Segmentation/feature_extraction/outputs/features_30fps/gt_arr'

    des_feat_dir = 'augmentation/swapped_videos'
    des_groundtruth_dir = 'augmentation/swapped_groundtruth'

    df_params = pd.read_csv(video_file, header=0)
    

    # with open(video_file) as file:
    #     video_paths = [line.rstrip() for line in file]
    for index, row in df_params.iterrows():
        video_path = row['path']
        idx1 = row['idx1']
        idx2 = row['idx2']
    #for video_path in video_paths:
        _, video_name = os.path.split(video_path)
        groundtruth = load_groundtruth(os.path.join(groundtruth_dir, video_name))
        features = load_feats(video_path)
        print(f"Features's shape before swapping {features.shape}")
        labels, starts, ends = get_segments(groundtruth=groundtruth)
        labels, starts, ends = list(map(int, labels)), list(map(int, starts)), list(map(int, ends))
        
        print(f"Labels before swapping: ", labels)
        new_labels, new_starts, new_ends = swap_segments(labels, starts, ends, idx1, idx2)

        #get last frame
        new_ends[-1] = max(new_ends[-1], features.shape[1])
        print(f"Labels after swapping: ", labels)
        
        transpose_features = np.transpose(features)
        swapped_features = transpose_features[new_starts[0]:new_ends[0]]

        swapped_groundtruth = groundtruth[new_starts[0]:new_ends[0]]
        #print(groundtruth[new_starts[0]:new_ends[0]])
        for i in range(1, len(new_labels)):
            #print(transpose_features[new_starts[i]:new_ends[i], :].shape)
            swapped_features = np.concatenate((swapped_features, transpose_features[new_starts[i]:new_ends[i], :]))
            swapped_groundtruth = np.concatenate((swapped_groundtruth, groundtruth[new_starts[i]:new_ends[i]]))
        swapped_features = np.transpose(swapped_features)

        print(f"Features's shape before swapping {swapped_features.shape}")
        print(swapped_features.shape)
        print(swapped_groundtruth.shape)

        np.save(os.path.join(des_feat_dir, video_name), swapped_features)
        np.save(os.path.join(des_groundtruth_dir, video_name), swapped_groundtruth)
