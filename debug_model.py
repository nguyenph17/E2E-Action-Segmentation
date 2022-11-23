import numpy as np
import pandas as pd
import os
from visualization.plot_segments import PlotSegments
import glob
from visualization import metrics
from matplotlib import pyplot as plt






def visualize(result_path, video_path):
    base = os.path.basename(video_path)
    video_name = os.path.splitext(base)[0]

    gt_path = os.path.join(result_path, f'predictions/{video_name}_gt.npy')
    #gt_path = './result_111122/new_result/predictions/{}_refined_pred.npy'.format(video_name)
    pred_path = os.path.join(result_path, f'predictions/{video_name}_refined_pred.npy')

    plot_segments = PlotSegments('./visualization/mapping.txt')

    plot_segments.visualize(video_path ,gt_path, pred_path)


def test_videos(result_path):
    pred_path = os.path.join(result_path, 'predictions')
    gt_format = '*_gt.npy'

    gt_files = [name for name in glob.glob(os.path.join(pred_path, gt_format))]

    scores = []
    print(f'There are {len(gt_files)} test files')
    for gt_file in gt_files:

        gt_file_name = os.path.split(gt_file)[-1]

        gt_file_name = str(gt_file_name).replace('_gt.npy', '')



        pred_file = os.path.join(pred_path, gt_file_name +'_refined_pred.npy')

        acc, edit, f1s = metrics.main(gt_file, pred_file)

        scores.append(list([acc, edit, f1s[-1]]))
        #print(f'file name: {gt_file_name}, scores: acc - {round(acc, 2)}, \
        #    edit - {round(edit, 2)}        f1@50 - {round(f1s[-1], 2)}')

    #print(scores)
    scores = np.array(scores)

    scores = np.sum(scores, axis=0) / scores.shape[0] * 1.0

    print(f'The mean score: acc={scores[0]}, edit={scores[1]}, f1@50={scores[2]}')
    #print(scores)

    return scores


def plot_training_loss(result_path):
    history = pd.read_csv(os.path.join(result_path, 'log.csv'))
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

def plot_val_scores(result_path):
    history = pd.read_csv(os.path.join(result_path, 'log.csv'))
    plt.plot(history['cls_acc'])
    plt.plot(history['edit'])
    plt.plot(history['segment f1s@0.5'])
    plt.title('model scores')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'edit', 'f1s@0.5'], loc='lower right')
    plt.show()



video_path = 'E:/AICamp/Human-Action-Reconigtion-Comparison/rgb/rgb/rgb-03-2.avi'
result_reduce_fps_aug = 'model_output/result_reduce_fps_aug'
new_result_agument = 'model_output/new_result_agument'
#plot_val_scores(result_path)
#visualize(result_path, video_path)
test_videos(new_result_agument)
test_videos(result_reduce_fps_aug)