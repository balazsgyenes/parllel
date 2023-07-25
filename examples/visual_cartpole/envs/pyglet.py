def set_default_pyglet_options(headless=False):
    import pyglet
    pyglet.options["headless"] = headless
    pyglet.options["vsync"] = False
    pyglet.options["xsync"] = False
