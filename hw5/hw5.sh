data_dir="$1"
output_dir="$2"

if [ ! -d "$output_dir" ]; then mkdir "$output_dir"; fi
python3 main.py "$data_dir" model/model2_no_norm.h5 -g 0 -o "$output_dir"
python3 confusion_matrix.py model/model2_no_norm-2.h5 -T -g 0 -o "$output_dir"