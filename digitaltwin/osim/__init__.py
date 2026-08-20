"""Optional OpenSim domain with lazy runtime imports."""

_EXPORTS = {
    "OpenSimModel": (".realtime.osim_model", "OpenSimModel"),
    "MuscleStateManager": (".realtime.muscle_state", "MuscleStateManager"),
}


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module
    module_name, attr_name = target
    try:
        value = getattr(import_module(module_name, __name__), attr_name)
    except ModuleNotFoundError as exc:
        if exc.name == "opensim":
            raise ImportError(
                "OpenSim support is optional. Install the OpenSim extra "
                "before importing digitaltwin.osim runtime classes."
            ) from None
        raise
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
