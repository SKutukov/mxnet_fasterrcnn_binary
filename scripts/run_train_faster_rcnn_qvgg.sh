step=$1
python3 train.py\
  --epochs 10 \
  --save-prefix \
  vgg_stage_0/fasterrcnn_vgg16_stage${step} \
  --step ${step} \
  --resume \
  /home/skutukov/Documents/model_test/fasterrcnn_vgg16_stage${step}-0000.params \
  --rcnn-batch-size 1 \
  --config_filename \
  configs/vgg/train/vgg_step_${step}.yml