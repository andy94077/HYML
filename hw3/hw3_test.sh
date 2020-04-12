data_dir="$1"
predicted_file="$2"
python3 main.py "$data_dir" model/test.h5 -eTs "$predicted_file.1.npy"
python3 main.py "$data_dir" model/test2.h5 -eTs "$predicted_file.2.npy"
python3 main.py "$data_dir" model/model2.h5 -f build_model2 -eTs "$predicted_file.3.npy"
python3 main.py "$data_dir" model/model2_no_norm.h5 -f build_model2 -eNTs "$predicted_file.4.npy"
python3 ensemble.py "$predicted_file" "$predicted_file.1.npy" "$predicted_file.2.npy" "$predicted_file.3.npy" "$predicted_file.4.npy"
rm -f "$predicted_file.1.npy" "$predicted_file.2.npy" "$predicted_file.3.npy" "$predicted_file.4.npy"

