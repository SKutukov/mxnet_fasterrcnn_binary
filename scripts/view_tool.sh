base_dir=/home/skutukov/Pictures/test_image/
full_pr_dir=$base_dir/result_full_precision
bin_pr_dir=$base_dir/result_full_bin

out_dir=/home/skutukov/Pictures/temp2

python3 ../tools/view_image.py \
  --full_precision_dir  $full_pr_dir \
  --binary_precision_dir $bin_pr_dir \
  --out_dir $out_dir
