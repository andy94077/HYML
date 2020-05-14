trainX="$1"
model_path="$2"
predict_csv="$3"

python3 autoencoder.py "$model_path" -t "$trainX" -f build_autoencoder3 -g 0 -Ts "$trainX" "${trainX}_latent.1.npy"
python3 main.py model/clustering3_whiten "${trainX}_latent.1.npy" -t improved_transform4 -s "$predict_csv"
rm -f "${trainX}_latent.1.npy"