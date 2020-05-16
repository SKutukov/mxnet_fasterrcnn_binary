step=$1
python3 test.py \
  --params \
  /home/skutukov/Documents/model_test/fasterrcnn_vgg16_stage${step}-0000.params \
  --step ${step} \
  --prefix temp1/ \
  --dataset voc \
  --config_filename configs/vgg/test/vgg_step_${step}.yml
