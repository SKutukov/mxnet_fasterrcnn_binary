#!/usr/bin/env bash
export PYTHONPATH=PYTHONPATH:./

rm speed.csv

for i in $(seq 1 100); do
  rm log.txt

  python3 test_speed_up/test_converter.py \
          --part 1 \
          --or_param_file \
          /home/skutukov/work/test_model/test-0000.params \
          --or_symbol_file \
          /home/skutukov/work/test_model/test-symbol.json \


  python3 test_speed_up/test_converter.py \
          --part 2 \
          --bin_param_file \
          /home/skutukov/work/test_model/binarized_test-0000.params \
          --bin_symbol_file \
          /home/skutukov/work/test_model/binarized_test-symbol.json

done
