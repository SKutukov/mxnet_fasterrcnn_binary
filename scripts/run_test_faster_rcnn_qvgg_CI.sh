step=$1

for i in {0..9};
 do
  imageset=2007_test-${i}
  echo $imageset

  python3 test.py \
    --params \
    /home/skutukov/work/mymx-rcnn/vgg16_PASCAL_step_${step}/-0037.params \
    --step ${step} \
    --prefix ${step}/CI_$i/ \
    --dataset voc \
    --config_filename configs/vgg/test/vgg_step_${step}.yml \
    --imageset $imageset \


done
