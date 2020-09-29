#Taken by https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/3

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

