#Taken by https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/3

class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))

    def __setitem__(self,name,value):
        return setattr(self.module,self.prefix + str(name),value)

    def __call__(self,i):

        return self.prefix + str(i)