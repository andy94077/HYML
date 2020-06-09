trainX="$1"
model_path="$2"

if [[ "${model_path##*/}" == "baseline.pth" ]]; then
	python3 autoencoder.py "$model_path" -g 0 -f build_baseline -t "$trainX"
else
	python3 autoencoder.py "$model_path" -g 0 -t "$trainX"
fi