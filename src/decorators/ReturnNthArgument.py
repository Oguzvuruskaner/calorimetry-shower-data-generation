from functools import wraps


def ReturnNthArgument(nth:int):

    def wrapper(func):

        @wraps(func)
        def inner_wrapper(*args,**kwargs):

            func(*args,**kwargs)
            #return self
            return args[nth]

        return inner_wrapper

    return wrapper