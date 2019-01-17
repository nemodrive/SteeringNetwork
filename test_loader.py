from data_loading import bddv_img_loader
from utils import config
import yaml

f = open("/home/ubuntu/nemodrive/SteeringNetwork/configs/bddv_img.yaml", 'r')
di = yaml.load(f)
ns = config.dict_to_namespace(di)

loader = bddv_img_loader.BDDVImageLoader(ns)
loader.load_data()
train_loader = loader.get_train_loader()
for i in enumerate(train_loader):
	print(i)