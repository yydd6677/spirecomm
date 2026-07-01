import enum


def to_primitive(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, list):
        return [to_primitive(item) for item in value]
    if isinstance(value, tuple):
        return [to_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        serialized = {}
        for key, item in value.__dict__.items():
            if not key.startswith("_"):
                serialized[key] = to_primitive(item)
        return serialized
    return repr(value)
