from functools import wraps
from importlib.util import find_spec


def module_is_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def depends_on_optional(module_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not module_is_available(module_name):
                raise ImportError(
                    f"Optional dependency {module_name} not found ({func.__name__})."
                )
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator
