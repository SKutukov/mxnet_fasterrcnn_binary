step=$1

for i in {0..50};
 do
  imageset=2007_test-${i}
  echo $imageset

  python3 test.py \
    --params \
    /home/skutukov/Documents/model_test/fasterrcnn_vgg16_stage${step}-0000.params \
    --step ${step} \
    --prefix ${step}/CI_$i/ \
    --dataset voc \
    --config_filename configs/vgg/test/vgg_step_${step}.yml \
    --imageset $imageset

done
