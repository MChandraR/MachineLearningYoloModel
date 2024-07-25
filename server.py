from flask import Flask, request
import core
import os

app = Flask("Cheat API")
# Tentukan direktori penyimpanan
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan direktori penyimpanan ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def index():
    core.run()
    return "Hallo"

@app.route("/detect", methods=['POST'])
def detect():
    try:
        imagefile = request.files.get('imagefile', '')
        if imagefile:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
            imagefile.save(filepath)
            return str(core.run(filepath))
    except Exception as err:
        return str(err)
    return "Normal tidak ada kecurangan"

app.run(debug=True)