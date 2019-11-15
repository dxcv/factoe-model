import json
import os
from pathlib import Path


class BasicConfig(object):
    def __init__(self, fp):
        with open(fp, 'rb') as f:
            config = json.load(f)
        for key, value in config.items():
            self.__dict__[key] = value


class Config(object):
    def __init__(self, fp, basic_fp):
        with open(basic_fp, 'rb') as f:
            basic_config = json.load(f)
        with open(fp, 'rb') as f:
            config = json.load(f)
        for key, value in config.items():
            self.__dict__[key] = value
        self.tmp_path = Path(self.tmp_path + '_' + self.flag)
        for key, value in basic_config.items():
            self.__dict__[key] = self.tmp_path / value
