trainX="$1"
model_path="$2"
predicted_file="$3"
invert=0
if [ "$4" == '-i' ]; then invert=1; fi

python3 autoencoder.py "$model_path.autoencoder.h5" -t "$trainX" -g 0
if [ "$invert" == 1 ]; then
	python3 main.py "$model_path" "$model_path.autoencoder.h5" -t "$trainX" -s "$predicted_file" -g 0 -i
else
	python3 main.py "$model_path" "$model_path.autoencoder.h5" -t "$trainX" -s "$predicted_file" -g 0
fi
