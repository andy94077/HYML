testX="$1"
predicted_file="$2"
gpu=0
python3 main.py model/word2vec model/lstm128.h5 -g $gpu -f build_model2 -eTs "$testX" "$predicted_file.1.npy"
python3 main.py model/word2vec model/lstm128-3.h5 -g $gpu -eTs "$testX" "$predicted_file.2.npy"
python3 main_acc.py model/word2vec model/lstm128-3_acc.h5 -g $gpu -eTs "$testX" "$predicted_file.3.npy"
python3 main.py model/word2vec model/lstm128_semi.h5 -g $gpu -eTs "$testX" "$predicted_file.4.npy"
python3 main.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128.h5 -g $gpu -f build_model2 -eTs "$testX" "$predicted_file.5.npy"
python3 main.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128-3.h5 -g $gpu -eTs "$testX" "$predicted_file.6.npy"
python3 main_acc.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128-3_acc.h5 -g $gpu -eTs "$testX" "$predicted_file.7.npy"
python3 main.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128_semi.h5 -g $gpu  -eTs "$testX" "$predicted_file.8.npy"
python3 main.py model/word2vec300 model/w2v_300_lstm128.h5 -g $gpu -f build_model2 -eTs "$testX" "$predicted_file.9.npy"
python3 main.py model/word2vec300 model/w2v_300_lstm128-3.h5 -g $gpu -eTs "$testX" "$predicted_file.10.npy"
python3 main_acc.py model/word2vec300 model/w2v_300_lstm128-3_acc.h5 -g $gpu -eTs "$testX" "$predicted_file.11.npy"
python3 main.py model/word2vec300 model/w2v_300_lstm128_semi.h5 -g $gpu -eTs "$testX" "$predicted_file.12.npy"
python3 ensemble.py "$predicted_file" $(for i in {1..12}; do echo -n "$predicted_file.$i.npy "; done)
rm -f $(for i in {1..12}; do echo -n "$predicted_file.$i.npy "; done)

