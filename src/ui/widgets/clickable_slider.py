"""
可点击滑块控件
"""
from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import Qt, pyqtSignal, QEvent


class ClickableSlider(QSlider):
    """支持双击事件的滑块"""
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj is self and event.type() == QEvent.MouseButtonDblClick:
            self.doubleClicked.emit()
            return True
        return super().eventFilter(obj, event)
