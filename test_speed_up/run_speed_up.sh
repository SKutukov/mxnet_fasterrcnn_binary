#!/usr/bin/env bash
#now=$(date +"%m-%d-%y")
now="speed_up_test_scaled"
mkdir -p /home/skutukov/$now

#for block_part in
python3 test_converter.py \
        --part 1 \
        --or_param_file \
        /home/skutukov/work/test_model/test-0000.params \
        --or_symbol_file \
        /home/skutukov/work/test_model/test-symbol.json \


python3 test_converter.py \
			  --part 2 \
        --bin_param_file \
        /home/skutukov/work/test_model/binarized_test-0000.params \
        --bin_symbol_file \
        /home/skutukov/work/test_model/binarized_test-symbol.json
