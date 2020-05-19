import random

filename = '/home/skutukov/work/data/VOCdevkit/VOC2007/ImageSets/Main/test.txt'

for j in range(0, 5):
    with open(filename) as test_file:
        imgs = test_file.readlines()
        random.shuffle(imgs)
        print(len(imgs))

        chunks = []
        for i in range(10):
            chunk_size = int(len(imgs) / 10)
            chunk = imgs[i * chunk_size:(i + 1) * chunk_size]
            chunks.append(chunk)

        for i in range(10):
            out_filename = 'test-{}.txt'.format(i + 10 * j)

            with open(out_filename, 'w') as out_file:
                out_file.writelines(chunks[i])
