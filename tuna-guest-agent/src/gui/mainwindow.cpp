#include "mainwindow.h"

#include <QTabWidget>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QFrame>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QSystemTrayIcon>
#include <QIcon>
#include <QTextEdit>
#include <QTextDocument>
#include <QTableWidget>
#include <QHeaderView>
#include <QLineEdit>
#include <QProcess>
#include <QTimer>
#include <QSettings>
#include <QMessageBox>
#include <QCoreApplication>
#include <QFileInfo>
#include <QRegularExpression>
#include <QStringList>

namespace {
QLabel* makeTitle(const QString& text) {
    auto* l = new QLabel(text);
    QFont f = l->font();
    f.setPointSizeF(f.pointSizeF() + 6);
    f.setBold(true);
    l->setFont(f);
    return l;
}

QFrame* card(QWidget* content, QWidget* parent = nullptr) {
    auto* c = new QFrame(parent);
    c->setObjectName("Card");
    auto* lay = new QVBoxLayout(c);
    lay->setContentsMargins(16,16,16,16);
    lay->addWidget(content);
    return c;
}
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {
    settings_ = loadSettings();
    workloadProcess_ = new QProcess(this);
    requestTimeout_ = new QTimer(this);
    requestTimeout_->setSingleShot(true);
    connect(requestTimeout_, &QTimer::timeout, this, [this]() {
        if (workloadProcess_->state() == QProcess::Running) {
            requestTimedOut_ = true;
            workloadProcess_->kill();
        }
    });
    connect(workloadProcess_,
            qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
            this,
            [this](int exitCode, QProcess::ExitStatus exitStatus) {
                onWorkloadFinished(exitCode, static_cast<int>(exitStatus));
            });
    connect(workloadProcess_, &QProcess::errorOccurred, this,
            [this](QProcess::ProcessError error) {
                if (error == QProcess::FailedToStart) {
                    requestTimeout_->stop();
                    runWorkloadButton_->setEnabled(true);
                    runWorkloadHeaderButton_->setEnabled(true);
                    cancelWorkloadButton_->setEnabled(false);
                    statusLabel_->setText("Sample client could not be started");
                    logsText_->append("Error: tuna_sample_app executable was not found or could not start.");
                }
            });

    auto* tabs = new QTabWidget(this);
    tabs->addTab(buildDashboard(), "Dashboard");
    tabs->addTab(buildPerformance(), "Performance");
    tabs->addTab(buildProcesses(), "Processes");
    tabs->addTab(buildTunnel(), "Tunnel");
    tabs->addTab(buildLogs(), "Logs");
    setCentralWidget(tabs);
    resize(1180, 760);
    if (!settings_.serverHost.isEmpty()) {
        serverLabel_->setText("Configured server: " + settings_.serverHost + ":" + settings_.serverPort);
        statusLabel_->setText("Configuration loaded; no live session is open");
        tunnelStatusLabel_->setText("Ready for an RPC request");
    }

    buildMenus();
    buildTray();
}

QWidget* MainWindow::buildDashboard() {
    auto* root = new QWidget(this);
    auto* v = new QVBoxLayout(root);
    v->setContentsMargins(16,16,16,16);
    v->setSpacing(16);

    // Header card
    auto* header = new QWidget(root);
    auto* headerLayout = new QGridLayout(header);
    headerLayout->setContentsMargins(12,12,12,12);
    headerLayout->setHorizontalSpacing(24);
    headerLayout->setVerticalSpacing(8);

    statusLabel_ = makeTitle("Not connected");
    statusLabel_->setProperty("accent", true);
    serverLabel_ = new QLabel("Live connection is not integrated in this GUI prototype");
    uptimeLabel_ = new QLabel("Uptime\nN/A");
    tlsLabel_ = new QLabel("Security\nmTLS");
    protocolLabel_ = new QLabel("Protocol\ngRPC/TLS");
    runWorkloadHeaderButton_ = new QPushButton("Run sample workload");
    runWorkloadHeaderButton_->setMinimumWidth(220);
    connect(runWorkloadHeaderButton_, &QPushButton::clicked, this, &MainWindow::onRunWorkload);

    headerLayout->addWidget(statusLabel_, 0, 0, 1, 3);
    headerLayout->addWidget(serverLabel_, 1, 0, 1, 3);
    headerLayout->addWidget(uptimeLabel_, 2, 0);
    headerLayout->addWidget(tlsLabel_, 2, 1);
    headerLayout->addWidget(protocolLabel_, 2, 2);
    headerLayout->addWidget(runWorkloadHeaderButton_, 1, 4, 2, 1);

    v->addWidget(card(header));

    // Metrics row
    auto* row = new QHBoxLayout();
    row->setSpacing(16);

    auto* cpuBox = new QWidget(root);
    auto* cpuLay = new QVBoxLayout(cpuBox);
    cpuLay->addWidget(new QLabel("CPU Offload (not available)"));
    cpuBar_ = new QProgressBar(cpuBox);
    cpuBar_->setRange(0, 100);
    cpuBar_->setValue(0);
    cpuBar_->setFormat("N/A");
    cpuLay->addWidget(cpuBar_);

    auto* gpuBox = new QWidget(root);
    auto* gpuLay = new QVBoxLayout(gpuBox);
    gpuLay->addWidget(new QLabel("GPU Usage (not available)"));
    gpuBar_ = new QProgressBar(gpuBox);
    gpuBar_->setRange(0, 100);
    gpuBar_->setValue(0);
    gpuBar_->setFormat("N/A");
    gpuLay->addWidget(gpuBar_);

    auto* latBox = new QWidget(root);
    auto* latLay = new QVBoxLayout(latBox);
    latLay->addWidget(new QLabel("Tunnel Latency (not available)"));
    latencyBar_ = new QProgressBar(latBox);
    latencyBar_->setRange(0, 200);
    latencyBar_->setValue(0);
    latencyBar_->setFormat("N/A");
    latLay->addWidget(latencyBar_);

    row->addWidget(card(cpuBox));
    row->addWidget(card(gpuBox));
    row->addWidget(card(latBox));

    auto* rowWrap = new QWidget(root);
    rowWrap->setLayout(row);
    v->addWidget(rowWrap);

    auto* workloadCard = new QWidget(root);
    auto* workloadLayout = new QGridLayout(workloadCard);
    workloadLayout->addWidget(new QLabel("Sample remote workload: sum of squares", workloadCard), 0, 0, 1, 2);
    workloadInput_ = new QLineEdit(workloadCard);
    workloadInput_->setPlaceholderText("Enter 1 to 4096 unsigned integers, comma-separated (e.g. 3,4)");
    workloadLayout->addWidget(workloadInput_, 1, 0);
    runWorkloadButton_ = new QPushButton("Run on server", workloadCard);
    connect(runWorkloadButton_, &QPushButton::clicked, this, &MainWindow::onRunWorkload);
    workloadLayout->addWidget(runWorkloadButton_, 1, 1);
    cancelWorkloadButton_ = new QPushButton("Cancel", workloadCard);
    cancelWorkloadButton_->setEnabled(false);
    connect(cancelWorkloadButton_, &QPushButton::clicked, this, [this]() {
        if (workloadProcess_->state() == QProcess::Running) {
            requestCancelled_ = true;
            workloadProcess_->kill();
        }
    });
    workloadLayout->addWidget(cancelWorkloadButton_, 1, 2);
    auto* resultLabel = new QLabel("No remote request has been made.", workloadCard);
    resultLabel->setObjectName("workloadResult");
    workloadLayout->addWidget(resultLabel, 2, 0, 1, 2);
    v->addWidget(card(workloadCard));

    return root;
}

QWidget* MainWindow::buildPerformance() {
    auto* w = new QWidget(this);
    auto* l = new QVBoxLayout(w);
    auto* title = makeTitle("Metrics (no live client data)");
    l->addWidget(title);
    auto* table = new QTableWidget(0, 3, w);
    table->setHorizontalHeaderLabels({"Metric", "Current", "Trend"});
    table->horizontalHeader()->setStretchLastSection(true);
    l->addWidget(table);

    auto addRow = [table](const QString& name) {
        int r = table->rowCount();
        table->insertRow(r);
        table->setItem(r, 0, new QTableWidgetItem(name));
        table->setItem(r, 1, new QTableWidgetItem("N/A"));
        table->setItem(r, 2, new QTableWidgetItem("N/A"));
    };
    addRow("CPU Offload %");
    addRow("GPU Util %");
    addRow("Tunnel RTT ms");
    addRow("Throughput MB/s");

    return w;
}

QWidget* MainWindow::buildProcesses() {
    auto* w = new QWidget(this);
    auto* l = new QVBoxLayout(w);
    l->addWidget(makeTitle("Process forwarding is not implemented"));
    auto* table = new QTableWidget(0, 4, w);
    table->setHorizontalHeaderLabels({"Process", "PID", "Hooks", "Status"});
    table->horizontalHeader()->setStretchLastSection(true);
    l->addWidget(table);

    auto addProc = [table](const QString& name, int pid, const QString& hooks, const QString& status) {
        int r = table->rowCount();
        table->insertRow(r);
        table->setItem(r, 0, new QTableWidgetItem(name));
        table->setItem(r, 1, new QTableWidgetItem(QString::number(pid)));
        table->setItem(r, 2, new QTableWidgetItem(hooks));
        table->setItem(r, 3, new QTableWidgetItem(status));
    };
    addProc("N/A", 0, "No API hooks", "Unavailable");
    return w;
}

QWidget* MainWindow::buildTunnel() {
    auto* w = new QWidget(this);
    auto* l = new QVBoxLayout(w);
    l->addWidget(makeTitle("Tunnel Diagnostics"));
    auto* grid = new QGridLayout();
    l->addLayout(grid);

    grid->addWidget(new QLabel("Protocol:"), 0, 0);
    grid->addWidget(new QLabel("gRPC over TLS"), 0, 1);
    grid->addWidget(new QLabel("Cipher:"), 1, 0);
    grid->addWidget(new QLabel("mTLS configured by sample client"), 1, 1);
    grid->addWidget(new QLabel("Last RPC:"), 2, 0);
    tunnelStatusLabel_ = new QLabel("No request sent", w);
    grid->addWidget(tunnelStatusLabel_, 2, 1);
    grid->addWidget(new QLabel("Session ID:"), 3, 0);
    grid->addWidget(new QLabel("N/A"), 3, 1);

    auto* btnRow = new QHBoxLayout();
    auto* reconnect = new QPushButton("Reconnect");
    auto* safeMode = new QPushButton("Enable Safe Mode");
    reconnect->setEnabled(false);
    safeMode->setEnabled(false);
    btnRow->addWidget(reconnect);
    btnRow->addWidget(safeMode);
    btnRow->addStretch();
    l->addLayout(btnRow);
    return w;
}

QWidget* MainWindow::buildLogs() {
    auto* w = new QWidget(this);
    auto* l = new QVBoxLayout(w);
    l->addWidget(makeTitle("Agent Logs"));
    logsText_ = new QTextEdit(w);
    logsText_->setReadOnly(true);
    logsText_->document()->setMaximumBlockCount(1000);
    logsText_->setPlainText("No client RPC has been submitted.");
    l->addWidget(logsText_);
    return w;
}

void MainWindow::buildMenus() {
    auto* file = menuBar()->addMenu("File");
    auto* actExit = file->addAction("Exit");
    auto* conn = menuBar()->addMenu("Connection");
    auto* actSettings = conn->addAction("Settings...");
    auto* actRun = conn->addAction("Run sample workload");
    connect(actRun, &QAction::triggered, this, &MainWindow::onRunWorkload);
    auto* view = menuBar()->addMenu("View");
    auto* actLogs = view->addAction("Logs");

    connect(actExit, &QAction::triggered, this, [this]() { close(); });
    connect(actLogs, &QAction::triggered, this, [this]() {
        // Switch to Logs tab
        auto* tabs = findChild<QTabWidget*>();
        if (tabs) tabs->setCurrentIndex(4);
    });

    connect(actSettings, &QAction::triggered, this, [this]() {
        ClientSettings edited = settings_;
        if (openSettingsDialog(this, &edited)) {
            if (!saveSettings(edited)) {
                return;
            }
            settings_ = edited;
            serverLabel_->setText("Configured server: " + settings_.serverHost + ":" + settings_.serverPort);
            statusLabel_->setText("Configuration saved; no live session is open");
            tunnelStatusLabel_->setText("Ready for an RPC request");
            if (auto* resultLabel = findChild<QLabel*>("workloadResult")) {
                resultLabel->setText("No remote request has been made with this configuration.");
            }
        }
    });
}

void MainWindow::buildTray() {
    tray_ = new QSystemTrayIcon(QIcon(), this);
    tray_->setToolTip("Tuna Guest Agent");
    auto* menu = new QMenu(this);
    auto* actOpen = menu->addAction("Open");
    auto* actToggle = menu->addAction("Run sample workload");
    auto* actQuit = menu->addAction("Exit");
    connect(actOpen, &QAction::triggered, this, [this]() { showNormal(); activateWindow(); });
    connect(actToggle, &QAction::triggered, this, &MainWindow::onRunWorkload);
    connect(actQuit, &QAction::triggered, this, [this]() { close(); });
    tray_->setContextMenu(menu);
    tray_->show();
}

ClientSettings MainWindow::loadSettings() const {
    QSettings settings;
    return {
        settings.value("connection/serverHost").toString(),
        settings.value("connection/serverPort", "50051").toString(),
        settings.value("tls/serverCaFile").toString(),
        settings.value("tls/clientCertificateFile").toString(),
        settings.value("tls/clientPrivateKeyFile").toString(),
    };
}

bool MainWindow::saveSettings(const ClientSettings& settings) {
    QSettings values;
    values.setValue("connection/serverHost", settings.serverHost);
    values.setValue("connection/serverPort", settings.serverPort);
    values.setValue("tls/serverCaFile", settings.serverCaFile);
    values.setValue("tls/clientCertificateFile", settings.clientCertificateFile);
    values.setValue("tls/clientPrivateKeyFile", settings.clientPrivateKeyFile);
    values.sync();
    if (values.status() != QSettings::NoError) {
        QMessageBox::critical(this, "Settings could not be saved",
                              "The configuration could not be written to the local settings store.");
        return false;
    }
    return true;
}

void MainWindow::onRunWorkload() {
    if (workloadProcess_->state() != QProcess::NotRunning) {
        return;
    }
    if (settings_.serverHost.isEmpty() || settings_.serverPort.isEmpty() ||
        settings_.serverCaFile.isEmpty() || settings_.clientCertificateFile.isEmpty() ||
        settings_.clientPrivateKeyFile.isEmpty()) {
        ClientSettings edited = settings_;
        if (!openSettingsDialog(this, &edited)) {
            return;
        }
        if (!saveSettings(edited)) {
            return;
        }
        settings_ = edited;
        serverLabel_->setText("Configured server: " + settings_.serverHost + ":" + settings_.serverPort);
    }

    const QStringList inputParts = workloadInput_->text().split(',', Qt::KeepEmptyParts);
    if (inputParts.isEmpty() || inputParts.size() > 4096) {
        QMessageBox::warning(this, "Invalid workload",
                             "Enter between 1 and 4096 unsigned integers separated by commas.");
        return;
    }

    QStringList arguments{
        "--server", settings_.serverHost + ":" + settings_.serverPort,
        "--ca", settings_.serverCaFile,
        "--cert", settings_.clientCertificateFile,
        "--key", settings_.clientPrivateKeyFile,
    };
    for (const QString& part : inputParts) {
        bool valid = false;
        const QString token = part.trimmed();
        const qulonglong value = token.toULongLong(&valid);
        if (token.isEmpty() || !valid) {
            QMessageBox::warning(this, "Invalid workload",
                                 "Every value must be an unsigned 64-bit decimal integer.");
            return;
        }
        arguments.append(QString::number(value));
    }

#ifdef Q_OS_WIN
    const QString executable = QCoreApplication::applicationDirPath() + "/tuna_sample_app.exe";
#else
    const QString executable = QCoreApplication::applicationDirPath() + "/tuna_sample_app";
#endif
    if (!QFileInfo::exists(executable)) {
        QMessageBox::critical(this, "Sample client not found",
                              "tuna_sample_app must be installed beside the Tuna GUI.");
        return;
    }

    requestTimedOut_ = false;
    requestCancelled_ = false;
    runWorkloadButton_->setEnabled(false);
    runWorkloadHeaderButton_->setEnabled(false);
    cancelWorkloadButton_->setEnabled(true);
    statusLabel_->setText("Sending remote workload...");
    tunnelStatusLabel_->setText("RPC in progress");
    if (auto* resultLabel = findChild<QLabel*>("workloadResult")) {
        resultLabel->setText("Request in progress...");
    }
    logsText_->append("Submitting sample workload with " +
                      QString::number(inputParts.size()) + " values to " +
                      settings_.serverHost + ":" + settings_.serverPort);
    workloadProcess_->setProgram(executable);
    workloadProcess_->setArguments(arguments);
    workloadProcess_->start();
    requestTimeout_->start(12000);
}

void MainWindow::onWorkloadFinished(int exitCode, int exitStatus) {
    requestTimeout_->stop();
    runWorkloadButton_->setEnabled(true);
    runWorkloadHeaderButton_->setEnabled(true);
    cancelWorkloadButton_->setEnabled(false);
    const QString standardOutput = QString::fromLocal8Bit(workloadProcess_->readAllStandardOutput());
    const QString standardError = QString::fromLocal8Bit(workloadProcess_->readAllStandardError());
    if (requestTimedOut_) {
        statusLabel_->setText("Remote request timed out");
        tunnelStatusLabel_->setText("Timeout");
        if (auto* resultLabel = findChild<QLabel*>("workloadResult")) {
            resultLabel->setText("Remote request timed out.");
        }
        logsText_->append("Error: remote RPC exceeded the client timeout.");
        return;
    }
    if (requestCancelled_) {
        statusLabel_->setText("Remote request cancelled");
        tunnelStatusLabel_->setText("Cancelled");
        if (auto* resultLabel = findChild<QLabel*>("workloadResult")) {
            resultLabel->setText("Remote request cancelled.");
        }
        logsText_->append("Request cancelled by the user.");
        return;
    }
    if (exitCode != 0 || exitStatus != static_cast<int>(QProcess::NormalExit)) {
        statusLabel_->setText("Remote request failed");
        tunnelStatusLabel_->setText("RPC failed");
        if (auto* resultLabel = findChild<QLabel*>("workloadResult")) {
            resultLabel->setText("Remote request failed; see Logs for details.");
        }
        logsText_->append("Error: " + standardError.trimmed());
        return;
    }

    static const QRegularExpression resultPattern(R"(sum_of_squares=(\d+))");
    const auto match = resultPattern.match(standardOutput);
    if (!match.hasMatch()) {
        statusLabel_->setText("Invalid response from sample client");
        tunnelStatusLabel_->setText("RPC response invalid");
        if (auto* resultLabel = findChild<QLabel*>("workloadResult")) {
            resultLabel->setText("Sample client returned an invalid response.");
        }
        logsText_->append("Error: sample client returned no recognizable result.");
        return;
    }
    const QString result = match.captured(1);
    statusLabel_->setText("Last RPC succeeded");
    tunnelStatusLabel_->setText("Last RPC succeeded");
    if (auto* resultLabel = findChild<QLabel*>("workloadResult")) {
        resultLabel->setText("Remote result: " + result);
    }
    logsText_->append("RPC succeeded; remote result=" + result);
}
