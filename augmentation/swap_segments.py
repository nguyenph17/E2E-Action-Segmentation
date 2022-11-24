import numpy as np
from random import shuffle
import math
import cv2
import os 
import glob
import time

class SwapSegment:
    def __init__(self, video_dir, groundtruth_dir, new_video_dir, new_groundtruth_dir) -> None:
        self.video_dir = video_dir
        self.groundtruth_dir = groundtruth_dir
        self.new_video_dir = new_video_dir
        self.new_groundtruth_dir = new_groundtruth_dir


    def get_video_info(self, video_name):
        self.video_name =  video_name
        video_path = os.path.join(self.video_dir, self.video_name)
        self.cap = cv2.VideoCapture(video_path)

        self.VIDEO_WIDTH  = math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        self.VIDEO_HEIGHT = math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

        
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)

        self.total_frames =self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        return self.VIDEO_WIDTH, self.VIDEO_HEIGHT


    def get_frames(self):
        self.frames = []
        if (self.cap.isOpened()== False):
            print("Error opening video file")
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                self.frames.append(frame)
            else:
                break
        self.cap.release()
        return self.frames

        


    def load_labels(self, label_file):
        file_ext = label_file.split('.')[-1]
        if file_ext == 'npy':
            labels = np.load(label_file)
            np_arr = np.load(label_file)
            inv_mapping = {int(v): k for k, v in self.mapping.items()}
            self.labels = list(map(inv_mapping.get, np_arr))

        elif file_ext == 'txt':
            with open(os.path.join(self.groundtruth_dir, label_file)) as file:
                self.labels = [line.rstrip() for line in file]
        else:
            print('Incorrect file extension! Please double check the file!')
            self.labels = None
        
        return self.labels


    def get_segments(self, frame_wise_labels, bg_class=["background"]):
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


    def swap_labels(self, labels, starts, ends):
        starting_segment, starting_start, starting_end = labels.pop(0), starts.pop(0), ends.pop(0)
        ending_segment, ending_start, ending_end = labels.pop(-1), starts.pop(-1), ends.pop(-1)

        c = list(zip(labels, starts, ends))
        shuffle(c)
        new_segments, new_starts, new_ends  = zip(*c)

        shuffle(c)

        new_segments, new_starts, new_ends = list(new_segments), list(new_starts), list(new_ends)


        new_segments.insert(0, starting_segment)
        new_starts.insert(0, starting_start)
        new_ends.insert(0, starting_end)

        new_segments.append(ending_segment)
        new_starts.append(ending_start)
        new_ends.append(ending_end)

        assert len(new_segments) == len(new_starts) == len(new_ends), "The length of Segments, Starts and Ends are not equal!"


        return new_segments, new_starts, new_ends 


    def swap_video(self, new_starts: list, new_ends:list) -> list:
        self.new_frames = []
        self.new_labels = []

        for start, end in zip(new_starts, new_ends):
            self.new_frames = self.new_frames + self.frames[start: end]
            self.new_labels = self.new_labels + self.labels[start: end]


        self.total_frames = min(max(new_ends), int(self.total_frames))

        self.labels = self.labels[:self.total_frames]
        self.new_labels = self.new_labels[:self.total_frames]


        # assert len(self.frames) == len(self.new_frames), 'The lenght of frames before and after swapping are not match!'
        # assert len(self.labels) == len(self.new_labels), 'The lenght of frame-wise label before and after swapping are not match!'
        return self.new_frames


    def save_groundtruth(self):
        split_tup = os.path.splitext(self.video_name)
        gt_filename = os.path.join(self.new_groundtruth_dir, split_tup[0] + '.txt')
        with open(gt_filename, 'w') as f:
            for new_label in self.new_labels:
                f.write(new_label + '\n')
        return gt_filename

    
    def save_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(self.new_video_dir, self.video_name)
        out = cv2.VideoWriter(video_path,fourcc, 30.0, (self.VIDEO_WIDTH,self.VIDEO_HEIGHT))
        
        for frame in self.new_frames:
            out.write(frame)
        out.release()
        return video_path


    def process(self):
        for video_path in glob.glob(os.path.join(self.video_dir,'*.avi')):
            _, video_name = os.path.split(video_path)

            print('Start generating video ', video_name)
            start_time = time.time()
            self.get_video_info(video_name)
            self.get_frames()

            split_tup = os.path.splitext(self.video_name)
            label_file = split_tup[0] + '.txt'
            self.load_labels(label_file)
            segments, starts, ends = self.get_segments(self.labels)

            segments, starts, ends = self.swap_labels(segments, starts, ends)

            self.swap_video(starts, ends)

            self.save_groundtruth()
            self.save_video()

            print("--- Duration %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    swapper = SwapSegment(video_dir='visualization/videos', groundtruth_dir='visualization/groundtruth', 
                        new_video_dir='visualization/videos_swap_segments', 
                        new_groundtruth_dir='visualization/groundtruth_swap_segments')

    swapper.process()


