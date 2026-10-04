#include <QApplication>

#include "mainwindow.h"

namespace {

QString buildStyleSheet() {
    return R"(
        * { font-family: "Segoe UI", Arial, sans-serif; }
        QMainWindow { background: #0e1116; }
        QWidget { color: #e6f1ff; }
        QTabBar::tab {
            background: #141922; color: #a9c1d9; padding: 10px 16px;
            border: 0; margin-right: 8px; border-radius: 8px;
        }
        QTabBar::tab:selected { background: #182131; color: #d8e6f6; }
        QTabWidget::pane { border: 0; }
        QLabel[accent="true"] { color: #78e2ff; }
        QPushButton {
            background: #167da0; color: white; border: 0;
            border-radius: 8px; padding: 9px 16px;
        }
        QPushButton:hover { background: #1b98bd; }
        QPushButton:disabled { background: #283442; color: #8999aa; }
        QLineEdit, QTextEdit {
            background: #0b1017; border: 1px solid #2b3b4d;
            border-radius: 6px; padding: 8px; selection-background-color: #167da0;
        }
        QLineEdit:focus, QTextEdit:focus { border-color: #38b8df; }
        QFrame#Card { background: #0f1520; border: 1px solid #1e2a3a; border-radius: 12px; }
        QMenuBar { background: #0e1116; color: #a9c1d9; }
        QMenu { background: #0f1520; color: #d8e6f6; border: 1px solid #1e2a3a; }
        QMenu::item:selected { background: #182131; }
    )";
}

}

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("Tuna Guest Agent");
    QApplication::setOrganizationName("Tuna");
    app.setStyleSheet(buildStyleSheet());

    MainWindow window;
    window.show();
    return app.exec();
}
