#!/usr/bin/env bash
#now=$(date +"%m-%d-%y")
now="speed_up_test_scaled"
mkdir -p /home/skutukov/$now

#for block_part in
python3 test_converter1.py \
        --part 1 \
        --block_part full

python3 test_converter1.py \
			  --part 2 \
			  --block_part full
