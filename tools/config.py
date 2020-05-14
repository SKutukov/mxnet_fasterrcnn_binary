import yaml


class Config:
    def __init__(self, filename):
        with open(filename) as f:
            temp_dict = yaml.load(f.read())

        self.rpn = temp_dict['rpn']
        self.train_param = temp_dict['train_param']
        self.transform = temp_dict['transform']


if __name__ == '__main__':
    config = Config('configs/vgg_step_4.yml')
    pass
