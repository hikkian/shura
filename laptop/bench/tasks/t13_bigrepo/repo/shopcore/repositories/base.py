class BaseRepository:
    def __init__(self):
        self._data = {}

    def add(self, obj, key=None):
        key = key if key is not None else getattr(obj, "id")
        self._data[key] = obj
        return obj

    def get(self, key):
        return self._data.get(key)

    def all(self):
        return list(self._data.values())

    def delete(self, key):
        self._data.pop(key, None)

    def clear(self):
        self._data.clear()
