trainX="$1"
model_path="$2"

python3 autoencoder.py "$model_path" -t "$trainX" -g 0
