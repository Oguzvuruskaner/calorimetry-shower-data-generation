from src.transformers.ITransformation import ITransformation
from src.decorators.ReturnNthArgument import ReturnNthArgument


class ISideEffect(ITransformation):

    _registry = []

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args,**kwargs)

        if cls not in ISideEffect._registry:
            ISideEffect._registry.append(cls)
            cls.transform = ReturnNthArgument(1)(cls.transform)
