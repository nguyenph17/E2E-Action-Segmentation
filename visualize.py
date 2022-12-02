import cv2
import argparse
import visualization.plot_segments as plot_segments
import os
import sys
import numpy as np
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help="path to desire input video file")
    parser.add_argument('--gt_file', type=str, help="path to ground truth of that video file")
    parser.add_argument('--prediction_file', type=str, help="path to prediction npy of that video file")
    args = parser.parse_args()

    ground_truth_file = 'rgb-01-1.txt'
    prediction_file = 'rgb-01-2.txt'
    mapping_file = 'visualization/mapping.txt'
    video_path = 'E:/AICamp/Human-Action-Reconigtion-Comparison/rgb/rgb/rgb-01-2.avi'
    

    plot_segment = plot_segments.PlotSegments(mapping_file)
    plot_segment.visualize(args.video_path ,args.gt_file, args.prediction_file)
