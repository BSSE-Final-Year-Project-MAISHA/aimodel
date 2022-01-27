from flask import Flask, render_template, request, redirect

import numpy as np
import joblib
from pydub import AudioSegment
from pydub.utils import make_chunks
import librosa  
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def preprocess(audio_file):
	"""
	Creates mel-spectrogram features from audio data
	"""
	audio = AudioSegment(audio_file)
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

@app.route("/", methods=["GET", "POST"])
def index():
	results=""
	status=""
	model = load_model("model.h5")
	scaler = joblib.load('scaler.pkl')

	if request.method == "POST":

		if "file" not in request.files:
			return redirect(request.url)

		#Get user audio file and id
		file = request.files["file"]

		if file.filename == "":
			return redirect(request.url)

		if file:
			#Extract features and run inference on the audio file
			feature = preprocess(file)

			if feature.shape[2] != 469:
				feature = np.resize(feature, (feature.shape[0], 80, 469))

			for sample in range(feature.shape[0]):
				feature[sample] = scaler.transform(feature[sample])
			predictions = model.predict(feature)
			results = round(np.mean(predictions))

			if results <= 4:
				status = f'PHQ score is {results}. Your depression is minimal'
			elif results <= 9:
				status = f'PHQ score is {results}. Your depression is mild'
			elif results <= 14:
				status = f'PHQ score is {results}. Your depression is moderate'
			elif results <= 19:
				status = f'PHQ score is {results}. Your depression is moderately severe'
			else:
				status = f'PHQ score is {results}. Your depression is severe'

			return render_template('results.html', status=status)

	return render_template('index.html')


if __name__ == "__main__":
	app.run(debug=True, threaded=True)
