source /opt/miniconda3/bin/activate torch22

models=("OmniScaleCNN" "gMLP" "TSSequencer")
# tasks=("sp" "occ" "flow")
tasks=("")

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        modelname="${model}"
        python ./train_mixedchannel.py --modelname "${modelname}"
    done
done
