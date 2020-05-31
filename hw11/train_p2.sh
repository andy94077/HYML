data_dir="$1"
model_path="$2"
python3 main.py "$data_dir" model/p2 -g 0 -f SN.build_model
mv model/p2/$(ls model/p1/models/*.h5 | tail -n 1) "$model_path"
rm -rf model/p2