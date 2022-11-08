#!/bin/bash
FILES=("1_KrELECTRA_train_ep80_bs16_lr1e_5_AdamW.py" "3_KoELECTRA_train_ep70_bs16_lr1e_5_AdamP.py")

for x in "${FILES[@]}"
do
    python3 "${x}"
done