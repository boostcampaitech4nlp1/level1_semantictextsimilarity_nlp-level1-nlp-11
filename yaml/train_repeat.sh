CONFIGS=("base_config", "roberta_config", "electra_config")

for (( i=0; i<3; i++ ))
do
    python3 train.py \
        —config ${CONFIGS[$i]} \
done