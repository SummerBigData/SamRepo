from flask import Flask, request
import uuid
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

@app.route('/upload', methods=['POST'])
def upload():
    print('uploading')
    fname = uuid.uuid4() + '.jpg'
    request.files['img'].save(app.config['UPLOAD_FOLDER'] + '/' + fname)
    return 'all good', 200

if __name__ == '__main__':
    app.run(