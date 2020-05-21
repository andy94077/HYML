data_dir="$1"
python3 seq2seq.py "$data_dir" model/exp_decay -g 0 -a -t 0.9987356 -tm exponential