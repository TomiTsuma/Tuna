#pragma once

#include <QMainWindow>
#include "settings_dialog.h"

class QLabel;
class QProgressBar;
class QTabWidget;
class QPushButton;
class QSystemTrayIcon;
class QLineEdit;
class QTextEdit;
class QProcess;
class QTimer;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private:
    QWidget* buildDashboard();
    QWidget* buildPerformance();
    QWidget* buildProcesses();
    QWidget* buildTunnel();
    QWidget* buildLogs();
    void buildMenus();
    void buildTray();
    void onRunWorkload();
    void onWorkloadFinished(int exitCode, int exitStatus);
    bool saveSettings(const ClientSettings& settings);
    ClientSettings loadSettings() const;

    // Dashboard widgets
    QLabel* statusLabel_{};
    QLabel* serverLabel_{};
    QLabel* uptimeLabel_{};
    QLabel* tlsLabel_{};
    QLabel* protocolLabel_{};
    QProgressBar* cpuBar_{};
    QProgressBar* gpuBar_{};
    QProgressBar* latencyBar_{};
    QPushButton* runWorkloadHeaderButton_{};
    QPushButton* runWorkloadButton_{};
    QPushButton* cancelWorkloadButton_{};
    QLineEdit* workloadInput_{};
    QTextEdit* logsText_{};
    QLabel* tunnelStatusLabel_{};
    QProcess* workloadProcess_{};
    QTimer* requestTimeout_{};
    bool requestTimedOut_{};
    bool requestCancelled_{};
    ClientSettings settings_;
    QSystemTrayIcon* tray_{};
};
