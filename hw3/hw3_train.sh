data_dir="$1"
cur_dir="$(pwd)"
cd "$data_diri/validation"
for i in $(ls *.jpg); do mv "$i" "../training/${i%.jpg}_.jpg"; done
cd "$cur_dir"
python3 main.py "$data_dir" model/test.h5 -r 880301
python3 main.py "$data_dir" model/test2.h5 -r 1126
python3 main.py "$data_dir" model/model2.h5 -f build_model2 -lr 1e-3
python3 main.py "$data_dir" model/model2_no_norm.h5 -N -f build_model2 -lr 1e-3

