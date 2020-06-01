#!/usr/bin/env bash
export PYTHONPATH=PYTHONPATH:./

path_to_converter=$2/build/tools/binary_converter/model-converter

for file in $1/ssd512*.params;
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

  # aggregate speed ups of binarized nn
  python3 test_speed_up/calc_final_metrics_v2.py \
    --filename_with_or_time $DIR/$base-time_or.csv \
    --filename_with_bin_time $DIR/$base-time_bn.csv \
    --or_param_path $or_param_file \
    --binarized_param_path  $bin_param_file \
    --log_path $DIR/$base-result.txt

done