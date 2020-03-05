from flask import Flask, jsonify, request
from test_GreenwayDiarization_original import output_function
import json, os
app = Flask(__name__)

data_folder = '/Users/macmini/Desktop/data'

if not os.path.exists(data_folder):
    os.mkdir(data_folder)

# @app.route('/transcribe', methods = ['POST'])
# # def transcription():
# def transcription():
#     total_input = str(request.data)
#     input_file = total_input.split(',')[0].replace("'","").replace('"','')
#     input_folder = total_input.split(',')[1].replace("'","").replace('"','')
#     # input_file = str(input_file)
#     # input_folder = str(input_folder)
#     run_function = output_function(input_file, input_folder)

#     return 'Transcription Successful'

@app.route('/')
def home():
    return 'The app is live!'


@app.route('/test', methods=['POST'])
# def transcription():
def test():

    total_input = request.json
    input_file = total_input.get('file_name')
    input_folder = total_input.get('file_location')

    return str(input_file)


@app.route('/transcribe', methods=['POST'])
# def transcription():
def transcription():
    total_input = request.json
    input_file = total_input.get('file_name')
    input_folder = total_input.get('file_location')
    run_function = output_function(input_file, input_folder)
    return 'transcription complete'

@app.route('/transcribe2', methods = ['POST'])
def transcription2():
    while not True:
        return 'Transcription in Progress'
    with open(request.json) as f:
        total_input = json.load(f)
        input_file = total_input.get('file_name')
        input_folder = total_input.get('file_location')
        run_function = output_function(input_file, input_folder)
        return 'transcription complete'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
