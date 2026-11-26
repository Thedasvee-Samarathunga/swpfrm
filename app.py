"""Swapframe entry point."""
import os
import sys

from PySide6.QtGui import QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from swapframe import __version__
from swapframe.ui.main_window import MainWindow


def resource(*parts: str) -> str:
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


def load_font() -> str:
    folder = resource('assets', 'fonts')
    family = None
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(('.ttf', '.otf')):
                fid = QFontDatabase.addApplicationFont(os.path.join(folder, name))
                fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                if fams and family is None:
                    family = fams[0]
    if family:
        return family
    return 'Segoe UI' if sys.platform == 'win32' else 'Sans Serif'


def _redirect_logs() -> None:
    """Windowed builds have no console; keep a log next to the models."""
    if sys.stderr is None or sys.stdout is None:
        from swapframe.models import models_dir
        log = open(os.path.join(os.path.dirname(models_dir()), 'swapframe.log'), 'a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


def main() -> int:
    _redirect_logs()
    smoke = '--smoke-test' in sys.argv
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Swapframe.App')
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName('Swapframe')
    app.setApplicationVersion(__version__)
    app.setStyle('Fusion')
    icon = resource('assets', 'icon.png')
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))

    win = MainWindow(load_font())   # applies the saved theme
    win.show()

    if smoke:  # CI: prove the bundle starts (Qt + onnxruntime + onnx + ffmpeg), then quit
        from PySide6.QtCore import QTimer
        import onnx  # noqa: F401
        from swapframe import media
        from swapframe.engine import available_devices
        ok = os.path.exists(media.ffmpeg_exe())
        print('devices:', available_devices(), 'ffmpeg:', ok, flush=True)
        QTimer.singleShot(2000, lambda: app.exit(0 if ok else 1))
    elif win.store.value('notice_ok', 'false') not in (True, 'true'):
        QMessageBox.information(
            win, 'Before you start',
            'Only swap faces of people who have agreed to it.\n\n'
            "Don't use Swapframe to deceive, harass or impersonate anyone, and label results as edited "
            'when you share them.')
        win.store.setValue('notice_ok', True)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
