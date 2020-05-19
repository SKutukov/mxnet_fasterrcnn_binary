step=$1

for i in {3..25};
 do
  imageset=2007_test-${i}
  echo $imageset

  python3 test.py \
    --params \
   /home/skutukov/Documents/mdel_resnet_test/fasterrcnn_resnet_stage${step}-0000.params \
    --step ${step} \
    --prefix ${step}/CI_$i/ \
    --dataset voc \
    --config_filename configs/resnet101/test/resnet101_step_${step}.yml \
    --imageset $imageset \
    --network resnet101

done
