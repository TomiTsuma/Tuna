#pragma once

#include <QMainWindow>

class QLabel;
class QProgressBar;
class QTabWidget;
class QPushButton;
class QSystemTrayIcon;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private slots:
    void onToggleConnection();

private:
    QWidget* buildDashboard();
    QWidget* buildPerformance();
    QWidget* buildProcesses();
    QWidget* buildTunnel();
    QWidget* buildLogs();
    void buildMenus();
    void buildTray();

    // Dashboard widgets
    QLabel* statusLabel_{};
    QLabel* serverLabel_{};
    QLabel* uptimeLabel_{};
    QLabel* tlsLabel_{};
    QLabel* protocolLabel_{};
    QProgressBar* cpuBar_{};
    QProgressBar* gpuBar_{};
    QProgressBar* latencyBar_{};
    QPushButton* disconnectButton_{};
    bool connected_ = true;
    QSystemTrayIcon* tray_{};
};


