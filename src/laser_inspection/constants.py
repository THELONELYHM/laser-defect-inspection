"""Dataset class names and visualization defaults."""

CLASS_NAMES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)

CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (74, 144, 226),
    (0, 165, 255),
    (60, 179, 113),
    (214, 112, 218),
    (255, 191, 0),
    (80, 80, 255),
)


def normalize_class_name(name: str) -> str:
    """Normalize common variants found in NEU-DET VOC XML files."""
    normalized = name.strip().lower().replace(" ", "_")
    aliases = {
        "pitted-surface": "pitted_surface",
        "pitted_surface": "pitted_surface",
        "rolled_in_scale": "rolled-in_scale",
        "rolled-in-scale": "rolled-in_scale",
        "rolled-in_scale": "rolled-in_scale",
    }
    return aliases.get(normalized, normalized)

