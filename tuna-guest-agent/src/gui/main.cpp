#include <QApplication>
#include "mainwindow.h"

static QString buildDarkStyleSheet() {
    // A compact dark theme stylesheet matching the mockup aesthetic
    return R"(
        * { font-family: "Segoe UI", Arial, sans-serif; }
        QMainWindow { background-color: #0e1116; }
        QWidget { color: #e6f1ff; }
        QTabBar::tab {
            background: #141922; color: #a9c1d9; padding: 10px 16px; border: 0; margin-right: 8px;
            border-radius: 8px;
        }
        QTabBar::tab:selected { background: #182131; color: #d8e6f6; }
        QTabWidget::pane { border: 0; }
        QGroupBox { border: 1px solid #1e2a3a; border-radius: 12px; margin-top: 10px; background:#101722; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 4px 6px; color:#a9c1d9; }
        QLabel[accent="true"] { color: #78e2ff; }
        QPushButton { background: #e04b59; color: white; border: 0; border-radius: 10px; padding: 10px 18px; }
        QPushButton:hover { background: #ff5f6f; }
        QProgressBar { background: #0d121a; border: 1px solid #1e2a3a; border-radius: 8px; color: #cbd6e2; height: 14px; text-align: center; }
        QProgressBar::chunk { background-color: #1cc4ff; border-radius: 8px; }
        QFrame#Card { background:#0f1520; border:1px solid #1e2a3a; border-radius:16px; }
        QMenuBar { background: #0e1116; color: #a9c1d9; }
        QMenu { background: #0f1520; color: #d8e6f6; border: 1px solid #1e2a3a; }
        QMenu::item:selected { background: #182131; }
        QTableWidget { background: #0f1520; gridline-color: #1e2a3a; }
        QHeaderView::section { background: #141922; color: #a9c1d9; border: 0; padding: 6px; }
        QTextEdit { background: #0f1520; border: 1px solid #1e2a3a; }
    )";
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    QApplication::setQuitOnLastWindowClosed(false);
    QApplication::setApplicationName("Tuna Guest Agent");
    QApplication::setOrganizationName("Tuna");

    app.setStyleSheet(buildDarkStyleSheet());

    MainWindow w;
    w.show();
    return app.exec();
}


