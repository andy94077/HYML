trainX="$1"
trainX_no_label="$2"
gpu=0
python3 main.py model/word2vec model/lstm128.h5 -g $gpu -f build_model2 -t "$trainX" "$trainX_no_label"
python3 main.py model/word2vec model/lstm128-3.h5 -g $gpu -t "$trainX" "$trainX_no_label"
python3 main_acc.py model/word2vec model/lstm128-3_acc.h5 -g $gpu -t "$trainX" "$trainX_no_label"
python3 main.py model/word2vec model/lstm128_semi.h5 -g $gpu -t "$trainX" "$trainX_no_label" --semi
python3 main.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128.h5 -g $gpu -f build_model2 -t "$trainX" "$trainX_no_label"
python3 main.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128-3.h5 -g $gpu -t "$trainX" "$trainX_no_label"
python3 main_acc.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128-3_acc.h5 -g $gpu -t "$trainX" "$trainX_no_label"
python3 main.py model/word2vec300_iter_10 model/w2v_300_iter_10_lstm128_semi.h5 -g $gpu -t "$trainX" "$trainX_no_label" --semi
python3 main.py model/word2vec300 model/w2v_300_lstm128.h5 -g $gpu -f build_model2 -t "$trainX" "$trainX_no_label"
python3 main.py model/word2vec300 model/w2v_300_lstm128-3.h5 -g $gpu -t "$trainX" "$trainX_no_label"
python3 main_acc.py model/word2vec300 model/w2v_300_lstm128-3_acc.h5 -g $gpu -t "$trainX" "$trainX_no_label"
python3 main.py model/word2vec300 model/w2v_300_lstm128_semi.h5 -g $gpu -t "$trainX" "$trainX_no_label" --semi
