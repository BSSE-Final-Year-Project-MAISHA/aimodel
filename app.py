from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin

import numpy as np
import joblib
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa  
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import mysql.connector
import mysql.connector.pooling


def preprocess(audio_file):
	"""
	Creates mel-spectrogram features from audio data
	"""
	#audio = AudioSegment(audio_file)
	audio = AudioSegment.from_wav(audio_file)
	audio_chunks = make_chunks(audio, 15000)
	features = []

	for chunk in audio_chunks[:-1]:
		spectrogram = librosa.stft(np.array(chunk.get_array_of_samples(), dtype='float64'))
		spectrogram_magnitude, _ = librosa.magphase(spectrogram)
		mel_scale = librosa.feature.melspectrogram(S=spectrogram_magnitude, sr=chunk.frame_rate, n_mels=80)
		mel_spectrogram = librosa.amplitude_to_db(mel_scale, ref=np.min)
		features.append(mel_spectrogram)

	return np.asarray(features).astype('float64')


app = Flask(__name__)
CORS(app)

#Create a database connection pool
dbconfig = {"host":"us-cdbr-east-04.cleardb.com", "user":"b1f025aacd34d8", "password":"0b41da97", "database":"heroku_f4a69e59281e2c9"}
cnx = mysql.connector.connect(pool_name = "model_db_pool",pool_size = 3, **dbconfig)

@app.route("/", methods=["GET", "POST"])
@cross_origin()
def index():
	results=""
	model = load_model("model.h5")
	scaler = joblib.load('scaler.pkl')

	if request.method == "POST":

		if "file" not in request.files:
			return redirect(request.url)

		#Get user audio file and id
		file = request.files["file"]
		user_id = request.form["id"]

		if file.filename == "":
			return redirect(request.url)

		if file:
			#Extract features and run inference on the audio file
			feature = preprocess(file)
			for sample in range(feature.shape[0]):
				feature[sample] = scaler.transform(feature[sample])
			predictions = model.predict(feature)
			results = round(np.mean(predictions))

			#Get a connection from the database connection pool
			mydb = mysql.connector.connect(pool_name=cnx.pool_name)
			mycursor = mydb.cursor()

			#Insert the predicted PHQ score into the database
			sql = "INSERT INTO diagnosis_report (phq_score, users_user_id) VALUES (%s, %s)"
			val = (str(results), str(user_id))
			mycursor.execute(sql, val)
			mydb.commit()

			return render_template('results.html', results=results, id=user_id)

	return render_template('index.html')


if __name__ == "__main__":
	app.run(debug=True, threaded=True)
