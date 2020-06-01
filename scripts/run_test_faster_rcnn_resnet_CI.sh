step=$1

for i in {0..9};
 do
  imageset=2007_test-${i}
  echo $imageset

  python3 test.py \
    --params \
   resnet101_MNIST_step_${step}/-0023.params \
    --step ${step} \
    --prefix ${step}/CI_$i/ \
    --dataset voc \
    --config_filename configs/resnet101/test/resnet101_step_${step}.yml \
    --imageset $imageset \
    --network resnet101

done
