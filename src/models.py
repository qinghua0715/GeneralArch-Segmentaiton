import importlib
from torch import nn


class ModelRegistry:
    def __init__(self):
        self.models = {}

    def register(self, name, module_path, class_name):
        """
        注册模型
        :param name: 模型名称，例如 "Unet"
        :param module_path: 模型类所在的模块路径，例如 "Models.Unet"
        :param class_name: 模型类的名称，例如 "Unet"
        """
        self.models[name] = (module_path, class_name)

    def get(self, name, *args, **kwargs):
        """获取模型实例"""
        if name not in self.models:
            raise ValueError(f"Model {name} not registered")
        module_path, class_name = self.models[name]
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class(*args, **kwargs)


# 创建全局注册器实例
model_registry = ModelRegistry()

# 注册模型
"""
        注册模型
        :param name: 模型名称，例如 "Unet"
        :param module_path: 模型类所在的模块路径，例如 "Models.Unet"
        :param class_name: 模型类的名称，例如 "Unet"
"""
model_registry.register("Unet", "Models.Unet", "Unet")
# 如果有其他模型，可以继续注册，例如：
model_registry.register("S_Net", "Models.SNet", "S_Net")
