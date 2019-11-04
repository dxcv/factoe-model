import json


class Config(object):
    def __init__(self, fp):
        with open(fp, 'rb') as f:
            config = json.load(f)
        for key, value in config.items():
            self.__dict__[key] = value