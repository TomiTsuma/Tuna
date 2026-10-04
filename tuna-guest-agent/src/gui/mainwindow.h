#pragma once

#include <QMainWindow>
#include "settings_dialog.h"

class QLabel;
class QTabWidget;
class QPushButton;
class QLineEdit;
class QTextEdit;
class QProcess;
class QTimer;
class QWidget;
class QCloseEvent;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private:
    void closeEvent(QCloseEvent* event) override;
    QWidget* buildDashboard();
    QWidget* buildActivity();
    void buildMenus();
    void openSettings();
    void updateConfigurationDisplay();
    void appendActivity(const QString& message);
    bool configurationIsReady() const;
    void onRunWorkload();
    void onRunGpuWorkload();
    void startWorkload(bool gpuWorkload);
    void onWorkloadFinished(int exitCode, int exitStatus);
    bool saveSettings(const ClientSettings& settings);
    ClientSettings loadSettings() const;

    // Dashboard widgets
    QLabel* statusLabel_{};
    QLabel* serverLabel_{};
    QLabel* securityLabel_{};
    QLabel* resultLabel_{};
    QPushButton* configureButton_{};
    QPushButton* runWorkloadButton_{};
    QPushButton* runGpuWorkloadButton_{};
    QPushButton* cancelWorkloadButton_{};
    QLineEdit* workloadInput_{};
    QLineEdit* matrixSizeInput_{};
    QTextEdit* logsText_{};
    QProcess* workloadProcess_{};
    QTimer* requestTimeout_{};
    bool requestTimedOut_{};
    bool requestCancelled_{};
    bool requestIsGpu_{};
    ClientSettings settings_;
};
