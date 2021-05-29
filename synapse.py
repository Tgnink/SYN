import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
<<<<<<< HEAD
=======
import logging
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
import synapse_UI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow =synapse_UI.Ui_MainWindow()
    
    MainWindow.show()
    sys.exit(app.exec_())

    