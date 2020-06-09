testX="$1"
model_path="$2"
predict_csv="$3"

if [[ "${model_path##*/}" == "baseline.pth" ]]; then
	python3 autoencoder.py "$model_path" -g 0 -f build_baseline -Ts "$testX" "$predict_csv.latents.npy"
else
	python3 autoencoder.py "$model_path" -g 0 -Ts "$testX" "$predict_csv.latents.npy"
fi
python3 report_2_pca.py "$predict_csv.latents.npy" "$predict_csv"
rm -f "$predict_csv.latents.npy"