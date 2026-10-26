"""Start the experimental workbench without loading the simulation state."""
import sys
from PyQt5.QtWidgets import QApplication
from utils.config import configure_matplotlib
from ui.experimental_import import ExperimentalImportWindow


def main():
    configure_matplotlib()
    app = QApplication(sys.argv)
    window = ExperimentalImportWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
