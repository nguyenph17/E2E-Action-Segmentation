from flask import Flask, request, jsonify, render_template, flash, Response, redirect, url_for
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage    
from inference import get_arguments, inference_video, predict_file
from visualization.plot_segments import PlotSegments
from comparison.comparison import process

app = Flask(__name__)
app.secret_key = 'HienVX'

class Plot_result(PlotSegments):

    def set_window_size(self):
        self.WINDOW_HEIGH = self.VIDEO_HEIGHT + 70
        self.WINDOW_WIDTH = self.VIDEO_WIDTH + 400
        self.window = np.zeros((self.WINDOW_HEIGH, self.WINDOW_WIDTH,3), np.uint8)

    def plot_single_segment_bar(self):
        pred_img_bar, _, _ = self.create_segment_bar(self.pred_file)

        self.window[self.VIDEO_HEIGHT : self.VIDEO_HEIGHT + pred_img_bar.shape[0], :pred_img_bar.shape[1]] = pred_img_bar[:, :]

        cv2.putText(self.window, 'Prediction', (self.VIDEO_WIDTH + 10, self.VIDEO_HEIGHT + 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                   fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        return self.window

    def play_stream_video(self, video_path, gt_path, pred_path):
        self.gt_file = gt_path
        self.pred_file = pred_path
        self.video_dir, self.video_name = os.path.split(video_path)
        self.get_video_size(video_path)
        self.set_window_size()
        self.plot_legend()
        # self.get_metrics()
        while True:
                bollean , frame = self.cap.read()
                self.plot_single_segment_bar()
                self.window [:frame.shape[0], :frame.shape[1]] = frame
                self.plot_playing_progress()
                try:
                    ret, buffer = cv2.imencode('.jpg', self.window)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass
    
    
dataset_file = 'comparison/gt_file.txt'
plot_segments = Plot_result('./visualization/mapping.txt')
plot_segments_comparison = Plot_result('./visualization/mapping.txt')
# video_path = 'test_inference/rgb-01-1.avi'
# gt_path = 'F:/ActionSegment/new_result_agument/predictions/rgb-01-1_gt.npy'
# pred_path = 'F:/ActionSegment/new_result_agument/predictions/rgb-01-1_refined_pred.npy'


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/show', methods=['GET', 'POST'])
def show():
    if request.method == 'POST':
        return redirect(url_for('comparison'))
    return render_template('show.html')


@app.route('/comparison')
def comparison():
    cv2.waitKey(0)
    return render_template('comparison.html', video_name=top_name, score=scores[0][1])


@app.route('/video_feed')
def video_feed():
    return Response(plot_segments.play_stream_video(video_path, gt_path, pred_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/comparison_video')
def comparison_video():
    return Response(plot_segments_comparison.play_stream_video(video_compare_path, gt_path, pred__compare_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/", methods=['GET', 'POST'])
def test():
    global gt_path, pred_path, video_path, video_compare_path, pred__compare_path, top_name, scores
    if request.method == 'POST':
        if not request.files['npy_file']:
            flash('Please attach file for training', 'warning')
            return render_template("index.html")

        else:
            file_npy = request.files['npy_file']
            file_name = file_npy.filename.rsplit('.')[0]
            file_npy.save(f'samples/{file_name}.npy')
            pred_path = predict_file(f'samples/{file_name}.npy', 'samples/')
            gt_path = f'samples/gt/{file_name}_gt.npy'

            scores, top_name = process(pred_path, dataset_file)
            print(top_name)
            video_compare_path = f'samples/videos/{top_name}.avi'
            video_path = f'samples/videos/{file_name}.avi'
            pred__compare_path = f'samples/gt/{top_name}_gt.npy'
            return redirect(url_for('comparison'))


if __name__ == '__main__':
    app.run(debug=True)