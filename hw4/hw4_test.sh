testX="$1"
predicted_file="$2"
gpu=3
python3 main.py model/word2vec model/lstm128.h5 -g $gpu -f build_model2 -eTs "$testX" "$predicted_file.1.npy"
python3 main.py model/word2vec model/lstm128-3.h5 -g $gpu -eTs "$testX" "$predicted_file.2.npy"
python3 main_acc.py model/word2vec model/lstm128-3_acc.h5 -g $gpu -eTs "$testX" "$predicted_file.3.npy"
python3 ensemble.py "$predicted_file" "$predicted_file.1.npy" "$predicted_file.2.npy" "$predicted_file.3.npy"
rm -f "$predicted_file.1.npy" "$predicted_file.2.npy" "$predicted_file.3.npy"

