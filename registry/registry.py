class Registry:

    def __init__(self, name):
        self._name = name
        self._dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self._name}, items={list(self._dict)})'
        return format_str

    def __contains__(self, item):
        return item in self._dict

    def __getitem__(self, key):
        return self.get(key)

    @property
    def name(self):
        return self._name

    @property
    def registry_dict(self):
        return self._dict

    def get(self, key):
        result = self._dict.get(key, None)
        if result is None:
            raise KeyError(f'{key} is not in the {self._name} registry')
        return result

    def register_model(self, object_to_register):
        """Register a class.
        :param object_to_register: Class or function to be registered.
        """
        if not (isinstance(object_to_register, type) or callable(object_to_register)):
            raise TypeError(f'object must be a class or callable, but got {type(object_to_register)}')
        object_name = object_to_register.__name__
        if object_name in self._dict:
            raise KeyError(f'{object_name} is already registered in {self.name}')
        self._dict[object_name] = object_to_register
