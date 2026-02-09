_notify_callback = None

def set_notifier(callback):
    global _notify_callback
    _notify_callback = callback

def notify_camera_blocked():
    if _notify_callback:
        _notify_callback()
