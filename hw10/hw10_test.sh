testX="$1"
model_path="$2"
predict_csv="$3"

python3 autoencoder.py "$model_path" -g 4 -Ts "$testX" "$predict_csv.latents.npy"
python3 main.py model/clustering2 "$predict_csv.latents.npy" -t improved_transform2 -k 10 -s "$predict_csv"
rm -f "$predict_csv.latents.npy"