model_path="$1" # baseline2/models/35_19000.h5
output_img="$2"
python3 main.py data "$model_path" -g 0 -f baseline.build_model -Ts "$output_img" -r 880301