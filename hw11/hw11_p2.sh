model_path="$1" # model/SN2/models/099_55200.h5
output_img="$2"
python3 main.py data "$model_path" -g 0 -f SN.build_model -Ts "$output_img" -r 880531