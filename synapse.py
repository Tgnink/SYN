import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import logging
import synapse_UI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow =synapse_UI.Ui_MainWindow()
    
    MainWindow.show()
    sys.exit(app.exec_())

    