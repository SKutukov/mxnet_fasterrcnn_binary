#!/usr/bin/env bash
export PYTHONPATH=PYTHONPATH:./

path_to_converter=$2/build/tools/binary_converter/model-converter

for file in $1/*.params;
 do
   filename=$(basename -- "$file")
   f="${filename%.*}"
   DIR=$(dirname "${file}")
   base="$(cut -d'-' -f1 <<<"$f")"

  or_param_file=$DIR/$filename
  or_symbol_file=$DIR/$base-symbol.json
  # remove old file
  rm speed.csv

  $path_to_converter $or_param_file

  bin_param_file=$DIR/binarized_$base-0000.params
  bin_symbol_file=$DIR/binarized_$base-symbol.json

  echo $bin_param_file
  echo $bin_symbol_file
  for i in $(seq 1 100); do
    # remove old file
    rm log.txt
    # calc time of original nn
    python3 test_speed_up/test_converter.py \
            --part 1 \
            --or_param_file $or_param_file \
            --or_symbol_file $or_symbol_file

    # calc time of binarized nn
    python3 test_speed_up/test_converter.py \
            --part 2 \
            --bin_param_file $bin_param_file \
            --bin_symbol_file $bin_symbol_file

  done

  # aggregate speed ups of binarized nn
  python3 test_speed_up/calc_final_metrics.py \
    --filename_with_speed speed.csv \
    --or_param_path $or_param_file \
    --binarized_param_path  $bin_param_file \
    --log_path $DIR/$base-result.txt

  cp speed.csv $DIR/$base-speed.csv
done