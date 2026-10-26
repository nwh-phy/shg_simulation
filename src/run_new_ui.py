"""
新UI测试入口
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication
from ui.main_window import NewMainWindow
from utils.config import configure_matplotlib


def main():
    configure_matplotlib()
    app = QApplication(sys.argv)
    window = NewMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
