input_dir="$1"
output_dir="$2"

python3 fgsm.py "$input_dir" "$output_dir" -g 0