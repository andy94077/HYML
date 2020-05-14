trainX="$1"
model_path="$2"

python3 autoencoder.py "$model_path" -t "$trainX" -f build_autoencoder3 -e 100 -g 0
