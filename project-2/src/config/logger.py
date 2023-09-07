import logging 


def verbose_logger(func):
    def wrapper(*args, **kwargs):
        print(f"\nExecuting {func.__name__} function...")
        result = func(*args, **kwargs)
        print(f"Finished executing {func.__name__} function.")
        return result
    return wrapper


def class_verbose_logger(func):
    def wrapper(instance, *args, **kwargs):
        verbose = instance.verbose
        if verbose:
            print(f"Executing class function: {func.__name__}...")
        result = func(instance, *args, **kwargs)
        if verbose:
            print(f"Finished executing class function: {func.__name__}.")
        return result
    return wrapper