class DialectAppError(Exception):
    """Base exception for expected application failures."""


class AudioProcessingError(DialectAppError):
    pass


class ModelNotAvailableError(DialectAppError):
    pass


class UnsupportedClassifierError(DialectAppError):
    pass
