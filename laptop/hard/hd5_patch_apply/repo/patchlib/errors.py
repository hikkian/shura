class PatchError(Exception):
    pass


class ConflictError(PatchError):
    def __init__(self, message, hunk=None):
        super().__init__(message)
        self.hunk = hunk
