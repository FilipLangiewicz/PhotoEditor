import os
import sys
from PyQt6.QtWidgets import QApplication

from PhotoEditor import PhotoEditor


basedir = os.path.dirname(__file__)

try:
    from ctypes import windll
    myappid = 'combayns.PhotoEditor.1.0.0'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

def load_stylesheet(filename):
    with open(filename, "r") as file:
        return file.read()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoEditor()
    app.setStyleSheet(load_stylesheet(os.path.join(basedir, "styles", "style.qss")))
    window.show()
    sys.exit(app.exec())
