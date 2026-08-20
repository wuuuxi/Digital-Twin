"""Visualization namespace with lazy optional imports."""

_EXPORTS = {
    "CurvePlotter": (".plot_curves", "CurvePlotter"),
    "MetronomePlayer": (".audio", "MetronomePlayer"),
    "AudioCueManager": (".audio", "AudioCueManager"),
    "GlobalAudioScheduler": (".realtime", "GlobalAudioScheduler"),
    "SpeedController": (".realtime", "SpeedController"),
    "plot_activation_3d": (".heatmap", "plot_activation_3d"),
    "compare_activation_maps": (".heatmap", "compare_activation_maps"),
    "draw_heatmap_2d": (".heatmap", "draw_heatmap_2d"),
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
        if exc.name == "pygame":
            raise ImportError(
                "Realtime audio visualization requires the realtime extra "
                "(pygame); offline plotting remains available without it."
            ) from None
        raise
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
