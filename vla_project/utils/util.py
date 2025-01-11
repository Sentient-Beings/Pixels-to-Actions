'''
    Utilities for the VLA project
'''
class AttributeDict(dict):
    """
    A dictionary subclass that allows attribute access as attributes
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()})"
    
    def __dir__(self):
        return list(super().__dir__()) + list(self.keys())
    
    def __copy__(self):
        return AttributeDict(super().copy())
    
    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
    
    def clear(self):
        super().clear()