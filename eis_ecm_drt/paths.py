import os
from datetime import datetime


def _windows_desktop_dir():
    try:
        import winreg

        key_name = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_name) as key:
            value, _ = winreg.QueryValueEx(key, "Desktop")
        return os.path.expandvars(value)
    except (ImportError, OSError):
        return None


def desktop_dir():
    if os.name == "nt":
        value = _windows_desktop_dir()
        if value:
            return os.path.abspath(value)

    home = os.path.expanduser("~")
    candidate = os.path.join(home, "Desktop")
    return candidate if os.path.isdir(candidate) else home


def analysis_root_dir():
    root = os.path.join(desktop_dir(), "EIS Analysis output")
    os.makedirs(root, exist_ok=True)
    return root


def new_analysis_output_dir(prefix="analysis"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(analysis_root_dir(), "%s_%s" % (prefix, timestamp))
    path = base
    index = 2
    while os.path.exists(path):
        path = "%s_%02d" % (base, index)
        index += 1
    os.makedirs(path)
    return path


def ensure_output_dir(output_dir=None, prefix="analysis"):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    return new_analysis_output_dir(prefix=prefix)
