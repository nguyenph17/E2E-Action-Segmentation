import cv2
import numpy as np
import os
import math
import datetime
import visualization.metrics as metrics



class PlotSegments():

    def __init__(self, mapping_file='./visualization/mapping.txt', BAR_HEIGHT=50,  BAR_WIDTH = 640, WINDOW_HEIGH = 720, WINDOW_WIDTH = 1024):
        self.mapping_file = mapping_file
        self.gt_file = None
        self.pred_file = None
        self.mapping = {} # {'cut_tomato':0, 'place_tomato_into_bowl':1}
        self.segment_ids = []
        self.segment_labels = []
        self.segment_colors = {} #{0: red, 1: green}

        self.BAR_HEIGHT = BAR_HEIGHT
        self.BAR_WIDTH = BAR_WIDTH

        self.WINDOW_HEIGH = WINDOW_HEIGH
        self.WINDOW_WIDTH = WINDOW_WIDTH

        self.window = np.zeros((WINDOW_HEIGH, WINDOW_WIDTH,3), np.uint8)

        self.load_mapping()
        self.generate_colors()


    def load_mapping(self):
        with open(self.mapping_file) as file:
            for line in file:
                values = line.split(' ')
                self.mapping[values[1].rstrip()] = values[0]
        
        self.segment_ids = list(self.mapping.values())
        self.segment_labels = list(self.mapping.keys())
        return self.mapping



    def load_labels(self, label_file):
        file_ext = label_file.split('.')[-1]
        if file_ext == 'npy':
            np_arr = np.load(label_file)
            inv_mapping = {int(v): k for k, v in self.mapping.items()}
            labels = list(map(inv_mapping.get, np_arr))

        elif file_ext == 'txt':
            with open(label_file) as file:
                labels = [line.rstrip() for line in file]
        else:
            print('Incorrect file extension! Please double check the file!')
            labels = None
        return labels



    def generate_colors(self):
        self.load_mapping()
        segment_len = len(self.segment_ids)
        for i in range(segment_len):
            if i <= math.ceil(segment_len / 3):
                if i % 3 == 0:
                    color = (int(i * math.floor(255.0 / (segment_len / 3.0))), 255, 255)
                elif i % 3 == 1:
                    color = (int(i * math.floor(255.0 / (segment_len / 3.0))), 0, 255)
                else:
                    color = (int(i * math.floor(255.0 / (segment_len / 3.0))), 255, 0)

            elif i > math.ceil(segment_len / 3) and i <= math.ceil(2*segment_len / 3):
                if i % 3 == 0:
                    color = (128, int((i % math.ceil(segment_len / 3)) * (255 / (segment_len / 3))), 255)
                elif i % 3 == 1:
                    color = (128, int((i % math.ceil(segment_len / 3)) * (255 / (segment_len / 3))), 0)
                else:
                    color = (0, int((i % math.ceil(segment_len / 3)) * (255 / (segment_len / 3))), 255)
            else:
                if i % 3 == 0:
                    color = (255, 255, int(i % math.ceil(2 * segment_len / 3.0) * (255.0 / (segment_len / 3.0))))
                elif i % 3 == 1:
                    color = (0, 255, int(i % math.ceil(2 * segment_len / 3.0) * (255.0 / (segment_len / 3.0))))
                else:
                    color = (255, 128, int(i % math.ceil(2 * segment_len / 3.0) * (255.0 / (segment_len / 3.0))))

            self.segment_colors[self.segment_ids[i]] = color
        
        return self.segment_colors



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


    def create_segment_bar(self, label_file, norm=True):
        gt_labels = self.load_labels(label_file)
        segments, starts, ends = self.get_segments(gt_labels)

        segment_widths = []
        for s, e in zip(starts, ends):
            if norm:
                segment_widths.append(int(round((e-s) * (self.BAR_WIDTH * 1.0 / len(gt_labels)), 0)))
            else: 
                segment_widths.append(e-s)
        
        segment_ids = list(map(self.mapping.get, segments))

        segment_colors = list(map(self.segment_colors.get, segment_ids))



        img_bar = np.zeros((self.BAR_HEIGHT, self.BAR_WIDTH,3), np.uint8)
        cursor = 0
        for i in range(len(segment_ids)):
            color = segment_colors[i]

            img_bar[:, cursor: cursor + segment_widths[i]] = color

            cursor += segment_widths[i]
        return img_bar, segment_ids, segment_widths

    
    def plot_legend(self, legend_height = 25, legend_width=50):
        i = 0
        
        for key, value in self.segment_colors.items():

            
            cv2.rectangle(self.window, (self.VIDEO_WIDTH, i ), (self.VIDEO_WIDTH + legend_width, i + legend_height) ,
                          value, -1)


            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (50, 50)
            
            # fontScale
            fontScale = 0.5
            
            # Blue color in BGR
            color = (255, 255, 255)
            
            # Line thickness of 2 px
            thickness = 1

            gt_label = [i for i in self.mapping if self.mapping[i]==key][0]

            # Using cv2.putText() method
            image = cv2.putText(self.window, str(gt_label), (self.VIDEO_WIDTH + legend_width + 10, i+ int(legend_height/2) + 5), font, 
                   fontScale, color, thickness, cv2.LINE_AA)

   

            i += 25
        
        return self.window



    def plot_segment_bar(self):
        gt_img_bar, _, _ = self.create_segment_bar(self.gt_file)
        pred_img_bar, _, _ = self.create_segment_bar(self.pred_file)

        #ground truth segment bar is bellow video
        self.window[self.VIDEO_HEIGHT : self.VIDEO_HEIGHT + gt_img_bar.shape[0], :gt_img_bar.shape[1]] = gt_img_bar[:, :]

        cv2.putText(self.window, 'Ground Truth', (self.VIDEO_WIDTH + 10, self.VIDEO_HEIGHT + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


        #prediction segment bar is bellow the ground truth segment bar
        self.window[self.VIDEO_HEIGHT + gt_img_bar.shape[0] + 10 : self.VIDEO_HEIGHT + pred_img_bar.shape[0] + 10 + pred_img_bar.shape[0], 0:pred_img_bar.shape[1]] = pred_img_bar[:, :]

        cv2.putText(self.window, 'Prediction', (self.VIDEO_WIDTH + 10, self.VIDEO_HEIGHT + gt_img_bar.shape[0] + 10 + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # self.plot_legend()

        # cv2.imshow('GFG', self.window)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return self.window


    
    def plot_playing_progress(self, width = 2,  height=110):
        #get next frame number out of all the frames for video
        nextFrameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        #get total number of frames in the video
        totalFrames =self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        x1 = math.ceil((nextFrameNo/totalFrames) * self.VIDEO_WIDTH)
        y1 = self.VIDEO_HEIGHT - 5

        y2 = self.VIDEO_HEIGHT + height
        
        
        #red line as progress on top of that
        #width is line thickness
        cv2.line(self.window, (x1, y1), (x1, y2), (0,0,0), width)
        return x1, y1, y2

    def get_video_size(self, video_path):
        # Create a VideoCapture object and read from input file
        self.cap = cv2.VideoCapture(video_path)

        self.VIDEO_WIDTH  = math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        self.VIDEO_HEIGHT = math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

        
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)

        return self.VIDEO_WIDTH, self.VIDEO_HEIGHT

    
    def set_window_size(self):

        self.WINDOW_HEIGH = self.VIDEO_HEIGHT + 120
        self.WINDOW_WIDTH = self.VIDEO_WIDTH + 400

        self.window = np.zeros((self.WINDOW_HEIGH, self.WINDOW_WIDTH,3), np.uint8)


    def show_other_info(self, acc=0.0, edit=0.0, f150=0.0 ):

        seconds = round(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)/ self.FPS)
        self.DURATION = seconds 

        cv2.putText(self.window, 'FPS: ' + str(int(self.FPS)) + '. Duration: ' + str(datetime.timedelta(seconds=seconds)) + ' seconds.', (math.ceil(self.VIDEO_WIDTH / 3), 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)


        score_location = (self.VIDEO_WIDTH + 150, self.VIDEO_HEIGHT + 30)

        cv2.putText(self.window, 'Accuracy    F1@50    Edit' , score_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        score_location = (self.VIDEO_WIDTH + 150, self.VIDEO_HEIGHT + 60)
        cv2.putText(self.window, str(round(acc, 2)) + '        ' + str(round(f150, 2))  +   '      ' + str(round(edit, 2))
                    , score_location, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)




    def play_video(self):

        # Check if camera opened successfully
        if (self.cap.isOpened()== False):
            print("Error opening video file")

        # Read until video is completed
        while(self.cap.isOpened()):
            
        # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
            # Display the resulting frame
                self.plot_segment_bar()
                self.window[:frame.shape[0], :frame.shape[1]] = frame
                self.plot_playing_progress()
                #acc, edit, f1s = metrics.main(self.gt_file, self.pred_file)
                #self.show_other_info(acc, edit, f1s[-1])

                title = 'Press Q to Exit!                        Video: ' + str(self.video_name) + \
                ' - Ground truth: ' + str(self.gt_file) + ' - Prediction: ' + str(self.pred_file)
                cv2.imshow(title, self.window)
                #print(frame.shape)
                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        # Break the loop
            else:
                break

        # When everything done, release
        # the video capture object
        self.cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

    
    def get_metrics(self):
        acc, edit, f1s = metrics.main(self.gt_file, self.pred_file)
        self.show_other_info(acc, edit, f1s[-1])




    def visualize(self, video_path, gt_path, pred_path):
        self.gt_file = gt_path
        self.pred_file = pred_path
        self.video_dir, self.video_name =  os.path.split(video_path)
        self.get_video_size(video_path)
        self.set_window_size()
        self.plot_legend()
        #self.get_metrics()
        self.play_video()







# if __name__ == '__main__':
#     ground_truth_file = 'rgb-01-1.txt'
#     prediction_file = 'rgb-01-2.txt'
#     mapping_file = 'mapping.txt'
#     video_path = 'E:/AICamp/Human-Action-Reconigtion-Comparison/rgb/rgb/rgb-01-2.avi'
    


#     plot_segments = PlotSegments(mapping_file)
#     plot_segments.visualize(video_path ,ground_truth_file, prediction_file)





