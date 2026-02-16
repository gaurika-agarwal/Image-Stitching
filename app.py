from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import os
import traceback
from werkzeug.utils import secure_filename
import panorama
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'supersecretkey'  # Required for flashing messages

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/download')
def download_image():
    try:
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'panorama_result.jpg')
        if os.path.exists(result_path):
            return send_file(result_path,
                           mimetype='image/jpeg',
                           as_attachment=True,
                           download_name='panorama_result.jpg')
        else:
            flash('No panorama image found')
            return redirect(url_for('home'))
    except Exception as e:
        flash(f'Error downloading image: {str(e)}')
        return redirect(url_for('home'))

@app.route('/stitch', methods=['POST'])
def stitch_images():
    try:
        if 'images' not in request.files:
            flash('No files uploaded')
            return render_template('index.html')
        
        files = request.files.getlist('images')
        
        if len(files) < 2:
            flash('Please upload at least 2 images')
            return render_template('index.html')
            
        # Save uploaded files
        filenames = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filenames.append(filepath)

        # Read images
        images = []
        for f in filenames:
            img = cv2.imread(f)
            if img is None:
                flash(f'Failed to read image: {os.path.basename(f)}')
                return render_template('index.html')
            # Ensure image is in uint8 format
            img = np.uint8(img)
            images.append(img)
        
        # Stitch images
        try:
            result = None
            for i in range(1, len(images)):
                if i == 1:
                    result = panorama.stitching(images[0], images[1])
                else:
                    result = panorama.stitching(result, images[i])
                    
            if result is None:
                flash('Failed to stitch images. Please ensure the images have enough matching features.')
                return render_template('index.html')
        except Exception as e:
            flash(f'Error while stitching images: {str(e)}')
            return render_template('index.html')

        if result is not None:
            # Save the result
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'panorama_result.jpg')
            cv2.imwrite(result_path, result)
            return render_template('result.html', result_image='uploads/panorama_result.jpg')
        else:
            flash('Failed to create panorama. Please try with different images.')
            return render_template('index.html')
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        flash('An error occurred while processing the images. Please try again.')
        return render_template('index.html')

    # Save uploaded files
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            filenames.append(filepath)

    # Read images
    images = [cv2.imread(f) for f in filenames]
    
    # Create stitcher object
    stitcher = Stitcher()
    
    # Stitch images
    result = None
    for i in range(1, len(images)):
        if i == 1:
            result, vis = stitcher.stitch([images[0], images[1]])
        else:
            result, vis = stitcher.stitch([result, images[i]])

    if result is not None:
        # Save the result
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'panorama_result.jpg')
        cv2.imwrite(result_path, result)
        return render_template('result.html', result_image='uploads/panorama_result.jpg')
    else:
        return 'Failed to create panorama', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)