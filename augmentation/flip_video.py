import math
import cv2
import numpy as np
import os
import time
import random
import glob

class VAugmentation():
    def __init__(self, des_dir) -> None:
        self.des_dir = des_dir

    def get_video_size(self, video_path):
        # Create a VideoCapture object and read from input file
        self.video_path = video_path
        self.video_dir, self.video_name =  os.path.split(video_path)

        self.cap = cv2.VideoCapture(video_path)

        self.VIDEO_WIDTH  = math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
        self.VIDEO_HEIGHT = math.ceil(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`

        
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)

        return self.VIDEO_WIDTH, self.VIDEO_HEIGHT

    
    def set_window_size(self):

        self.WINDOW_HEIGH = self.VIDEO_HEIGHT + 120
        self.WINDOW_WIDTH = self.VIDEO_WIDTH + 400

        self.window = np.zeros((self.WINDOW_HEIGH, self.WINDOW_WIDTH,3), np.uint8)


    def flip_video(self, flipMode=1, multiply_value = 1.0, play_video=True):
        """
        Flip videoo

        Args:
            flip_params (int, optional): 0: Vertical flip
                                        1: Horizontal flip
            . Defaults to 1.
        """
        flipped_video = []

        # Check if camera opened successfully
        if (self.cap.isOpened()== False):
            print("Error opening video file")

        totalFrames =self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        split_tup = os.path.splitext(self.video_name)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        new_file_name = split_tup[0] + '_augmented' + split_tup[1]
        new_file_path = os.path.join(self.des_dir, new_file_name)
        out = cv2.VideoWriter(new_file_path,fourcc, 30.0, (self.VIDEO_WIDTH,self.VIDEO_HEIGHT))

        print('Start generating video ', new_file_name)
        # Read until video is completed

        last_percent = 0.0
        while(self.cap.isOpened()):
            
        # Capture frame-by-frame
            ret, frame = self.cap.read()
            if ret == True:
            # Display the resulting frame
                frame = cv2.flip(frame, flipMode)

                image = frame.astype(np.float64)
                image *= multiply_value

                image = np.where(image > 255, 255, image)
                image = np.where(image < 0, 0, image)
                image = image.astype(np.uint8)


                out.write(image)


                flipped_video.append(image)

                title = 'Press Q to Exit!                        Video: ' + str(self.video_name) 
                nextFrameNo = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                #get total number of frames in the video
                
                current_percent = round((nextFrameNo / totalFrames * 1.0) * 100, 1)
                if current_percent >= last_percent + 10:
                    last_percent = current_percent
                    print(f'{current_percent}% - Processing video {self.video_name}')

                if play_video:
                    cv2.imshow(title, image)

                    #print(frame.shape)
                    # Press Q on keyboard to exit
                    if (cv2.waitKey(25) & 0xFF == ord('q')) & nextFrameNo == totalFrames:
                        break
                else:
                    if nextFrameNo == totalFrames:
                        break

        # Break the loop
            else:
                break

        # When everything done, release
        # the video capture object
        self.cap.release()
        out.release()

        print('DONE! New video is generated, the file path is at: ', new_file_path)

        # Closes all the frames
        cv2.destroyAllWindows()

        return new_file_path




    def augment_video(self, video_path):
        self.get_video_size(video_path)
        self.set_window_size()
        #horizontal flip and make video darker
        bright_ness = random.choice([0.6, 0.7, 1.0])
        self.flip_video(flipMode=1, multiply_value=bright_ness, play_video=False) 



if __name__ == '__main__':
    video_file = 'augmentation/videos_to_flip.txt'
    destination_dir = 'augmentation/flipped_videos'
    with open(video_file) as file:
        video_paths = [line.rstrip() for line in file]

    
    #exist_video_paths = glob.glob('E:/AICamp/Human-Action-Reconigtion-Comparison/rgb/rgb/*.avi')

    for video_path in video_paths:
        vaug = VAugmentation(destination_dir)
        #print(name)
        start_time = time.time()
        vaug.augment_video(video_path)
        vaug = None
        print("--- Duration %s seconds ---" % (time.time() - start_time))
    
 
