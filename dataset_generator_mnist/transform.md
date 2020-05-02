1) dowload MNIST  dataset
https://www.kaggle.com/scolianni/mnistasjpg#trainingSet.tar.gz

2) extract MNIST dataset

3) run script
```
python transform.py \
--mnist_folder /home/skutukov/Documents/trainingSet \
--width 720 \
--height 540 \
--count 10 \
--out_path /home/skutukov/datasets/MNIST/result
```
