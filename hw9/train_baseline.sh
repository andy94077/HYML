trainX="$1"
model_path="$2"

python3 autoencoder.py "$model_path.autoencoder.h5" -t "$trainX" -g 0
python3 main.py "$model_path" "$model_path.autoencoder.h5" -t "$trainX" -g 0
