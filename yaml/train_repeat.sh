CONFIGS=(koelectra-Copy1 koelectra-Copy2 koelectra-Copy3 koelectra-Copy4 koelectra-Copy5 koelectra-Copy6 koelectra-Copy7 koelectra-Copy8 koelectra-Copy9 koelectra-Copy10)

for (( i=0; i<10; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]} \
    python3 inference.py --config ${CONFIGS[$i]} \   
done

