"""Optional system tray icon (Show Settings / Quit) for the persistent app,
backed by pystray + Pillow. Both are optional dependencies - the whole app
must work identically without them, just without a tray (see start()'s
None return and overlay_core.py's handling of it)."""

try:
    import pystray
    from PIL import Image, ImageDraw
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _make_icon_image():
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((2, 2, size - 2, size - 2), fill=(47, 111, 237, 255))
    draw.text((size // 2 - 11, size // 2 - 9), "VC", fill="white")
    return img


def start(on_show, on_quit):
    """Starts the tray icon on its own background thread and returns the
    running pystray.Icon (call .stop() on it to end it), or None if
    pystray/Pillow aren't installed. on_show/on_quit are called from the
    tray's own thread, not the Tk thread - callers must marshal back
    themselves (e.g. via root.after(0, ...))."""
    if not AVAILABLE:
        return None

    menu = pystray.Menu(
        pystray.MenuItem("Show Settings", lambda: on_show(), default=True),
        pystray.MenuItem("Quit", lambda: on_quit()),
    )
    icon = pystray.Icon("vc-translator", _make_icon_image(), "vc-translator", menu)
    icon.run_detached()
    return icon
