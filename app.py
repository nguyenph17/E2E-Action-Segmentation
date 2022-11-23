from flask import Flask, request, jsonify, render_template, flash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage    
from inference import get_arguments, inference_video, main

app = Flask(__name__)
app.secret_key = 'HienVX'

@app.route('/')
def hello():
    return 'Hello'


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        if not request.files['npy_file']:
            flash('Please attach file for training', 'warning')
            return render_template("index.html")
        else:
            file_npy = request.files['npy_file']
            file_name = file_npy.filename.rsplit('.')[0]
            file_npy.save(f'static/{file_name}.npy')
            main(f'static/{file_name}.npy')
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)