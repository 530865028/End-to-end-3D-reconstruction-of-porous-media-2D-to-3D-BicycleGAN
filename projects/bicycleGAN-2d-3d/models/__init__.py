import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    # Given the option --model [modelname],
    # the file "models/modelname_model.py"
    # will be imported.
    model_filename = "models." + model_name + "_model"

    # 这一句相当于 import models.bicycle_gan_model

    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    # 通过opt.model来找对应的model,如'bicycle_gan_model',另一个是'pix2pix_model'，都继承于BaseModel
    model = find_model_using_name(opt.model)
    # 实例化
    instance = model()
    # 网络初始化（包括网络结构）
    # 这里是bicycle_gan_model的函数
    instance.initialize(opt)
    print("model [%s] was created" % (instance.name()))
    return instance
