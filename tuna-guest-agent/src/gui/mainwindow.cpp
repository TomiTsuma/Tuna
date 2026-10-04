#include "mainwindow.h"

#include <QAction>
#include <QCloseEvent>
#include <QDateTime>
#include <QFont>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QTabWidget>
#include <QTextDocument>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QCoreApplication>
#include <QStringList>

namespace {

QLabel* makeTitle(const QString& text, QWidget* parent = nullptr) {
    auto* label = new QLabel(text, parent);
    QFont font = label->font();
    font.setPointSizeF(font.pointSizeF() + 6);
    font.setBold(true);
    label->setFont(font);
    return label;
}

QWidget* makeCard(QWidget* content, QWidget* parent = nullptr) {
    auto* frame = new QFrame(parent);
    frame->setObjectName("Card");
    auto* layout = new QVBoxLayout(frame);
    layout->setContentsMargins(20, 18, 20, 18);
    layout->addWidget(content);
    return frame;
}

}

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
    setWindowTitle("Tuna Client");
    setMinimumSize(700, 520);
    resize(900, 660);

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
                    cancelWorkloadButton_->setEnabled(false);
                    configureButton_->setEnabled(true);
                    statusLabel_->setText("Sample client could not be started");
                    resultLabel_->setText("The sample client was not found or could not start.");
                    appendActivity("Error: tuna_sample_app could not be started.");
                }
            });

    auto* tabs = new QTabWidget(this);
    tabs->addTab(buildDashboard(), "Workload");
    tabs->addTab(buildActivity(), "Activity");
    setCentralWidget(tabs);

    updateConfigurationDisplay();
    buildMenus();
}

void MainWindow::closeEvent(QCloseEvent* event) {
    if (workloadProcess_->state() == QProcess::Running) {
        const auto answer = QMessageBox::question(
            this, "Request in progress",
            "Exiting will cancel the local client. The server may already have completed the request. Exit anyway?",
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (answer != QMessageBox::Yes) {
            event->ignore();
            return;
        }
        requestCancelled_ = true;
        requestTimeout_->stop();
        workloadProcess_->kill();
        workloadProcess_->waitForFinished(1000);
    }
    QMainWindow::closeEvent(event);
}

QWidget* MainWindow::buildDashboard() {
    auto* root = new QWidget(this);
    auto* layout = new QVBoxLayout(root);
    layout->setContentsMargins(24, 24, 24, 24);
    layout->setSpacing(18);

    auto* intro = new QWidget(root);
    auto* introLayout = new QVBoxLayout(intro);
    introLayout->setContentsMargins(0, 0, 0, 0);
    introLayout->setSpacing(6);
    introLayout->addWidget(makeTitle("Tuna Client", intro));
    auto* subtitle = new QLabel(
        "Send the supported sample workload to a configured Tuna server.", intro);
    subtitle->setWordWrap(true);
    introLayout->addWidget(subtitle);
    layout->addWidget(intro);

    auto* connectionContent = new QWidget(root);
    auto* connectionLayout = new QHBoxLayout(connectionContent);
    connectionLayout->setContentsMargins(0, 0, 0, 0);
    auto* connectionInfo = new QVBoxLayout();
    auto* connectionTitle = new QLabel("Server configuration", connectionContent);
    QFont headingFont = connectionTitle->font();
    headingFont.setBold(true);
    connectionTitle->setFont(headingFont);
    connectionInfo->addWidget(connectionTitle);
    serverLabel_ = new QLabel(connectionContent);
    securityLabel_ = new QLabel(connectionContent);
    securityLabel_->setWordWrap(true);
    connectionInfo->addWidget(serverLabel_);
    connectionInfo->addWidget(securityLabel_);
    connectionLayout->addLayout(connectionInfo, 1);
    configureButton_ = new QPushButton("Configure...", connectionContent);
    connect(configureButton_, &QPushButton::clicked, this, &MainWindow::openSettings);
    connectionLayout->addWidget(configureButton_);
    layout->addWidget(makeCard(connectionContent, root));

    auto* workloadContent = new QWidget(root);
    auto* workloadLayout = new QVBoxLayout(workloadContent);
    workloadLayout->setContentsMargins(0, 0, 0, 0);
    workloadLayout->setSpacing(12);
    auto* workloadTitle = new QLabel("Sum of squares", workloadContent);
    workloadTitle->setFont(headingFont);
    workloadLayout->addWidget(workloadTitle);

    auto* explanation = new QLabel(
        "Enter 1–4096 unsigned 64-bit integers. The server returns the sum of their squares.",
        workloadContent);
    explanation->setWordWrap(true);
    workloadLayout->addWidget(explanation);

    workloadInput_ = new QLineEdit(workloadContent);
    workloadInput_->setPlaceholderText("For example: 3, 4, 10");
    workloadInput_->setMaxLength(100000);
    workloadInput_->setAccessibleName("Workload values");
    workloadLayout->addWidget(workloadInput_);

    auto* actions = new QHBoxLayout();
    actions->addStretch();
    cancelWorkloadButton_ = new QPushButton("Cancel request", workloadContent);
    cancelWorkloadButton_->setEnabled(false);
    connect(cancelWorkloadButton_, &QPushButton::clicked, this, [this]() {
        if (workloadProcess_->state() == QProcess::Running) {
            requestCancelled_ = true;
            workloadProcess_->kill();
        }
    });
    actions->addWidget(cancelWorkloadButton_);
    runWorkloadButton_ = new QPushButton("Run on server", workloadContent);
    runWorkloadButton_->setDefault(true);
    connect(runWorkloadButton_, &QPushButton::clicked, this, &MainWindow::onRunWorkload);
    actions->addWidget(runWorkloadButton_);
    workloadLayout->addLayout(actions);

    statusLabel_ = new QLabel("Configure a server to begin.", workloadContent);
    statusLabel_->setObjectName("RequestStatus");
    statusLabel_->setProperty("accent", true);
    statusLabel_->setWordWrap(true);
    workloadLayout->addWidget(statusLabel_);
    resultLabel_ = new QLabel("No request has been submitted.", workloadContent);
    resultLabel_->setObjectName("WorkloadResult");
    resultLabel_->setWordWrap(true);
    workloadLayout->addWidget(resultLabel_);
    layout->addWidget(makeCard(workloadContent, root));
    layout->addStretch();
    return root;
}

QWidget* MainWindow::buildActivity() {
    auto* page = new QWidget(this);
    auto* layout = new QVBoxLayout(page);
    layout->setContentsMargins(24, 24, 24, 24);
    layout->setSpacing(12);

    auto* header = new QHBoxLayout();
    header->addWidget(makeTitle("Recent activity", page));
    header->addStretch();
    auto* clearButton = new QPushButton("Clear", page);
    connect(clearButton, &QPushButton::clicked, this, [this]() {
        logsText_->clear();
    });
    header->addWidget(clearButton);
    layout->addLayout(header);

    auto* note = new QLabel(
        "Activity from this GUI session only. It is not a persistent service or server log.",
        page);
    note->setWordWrap(true);
    layout->addWidget(note);
    logsText_ = new QTextEdit(page);
    logsText_->setReadOnly(true);
    logsText_->document()->setMaximumBlockCount(1000);
    logsText_->setPlaceholderText("Request activity will appear here.");
    layout->addWidget(logsText_, 1);
    return page;
}

void MainWindow::buildMenus() {
    auto* fileMenu = menuBar()->addMenu("File");
    auto* settingsAction = fileMenu->addAction("Connection settings...");
    fileMenu->addSeparator();
    auto* exitAction = fileMenu->addAction("Exit");
    connect(settingsAction, &QAction::triggered, this, &MainWindow::openSettings);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    auto* workloadMenu = menuBar()->addMenu("Workload");
    auto* runAction = workloadMenu->addAction("Run sample workload");
    connect(runAction, &QAction::triggered, this, &MainWindow::onRunWorkload);

    auto* viewMenu = menuBar()->addMenu("View");
    auto* activityAction = viewMenu->addAction("Recent activity");
    connect(activityAction, &QAction::triggered, this, [this]() {
        if (auto* tabs = findChild<QTabWidget*>()) {
            tabs->setCurrentIndex(1);
        }
    });
}

void MainWindow::openSettings() {
    if (workloadProcess_->state() != QProcess::NotRunning) {
        QMessageBox::information(this, "Request in progress",
                                 "Wait for or cancel the current request before changing its connection settings.");
        return;
    }
    ClientSettings edited = settings_;
    if (!openSettingsDialog(this, &edited)) {
        return;
    }
    if (!saveSettings(edited)) {
        return;
    }
    settings_ = edited;
    updateConfigurationDisplay();
    resultLabel_->setText("No request has been submitted with this configuration.");
    appendActivity("Connection settings saved for " + settings_.serverHost + ":" +
                   settings_.serverPort + ".");
}

void MainWindow::updateConfigurationDisplay() {
    if (settings_.serverHost.isEmpty()) {
        serverLabel_->setText("No server configured.");
        securityLabel_->setText("A server and administrator-provisioned mTLS credentials are required.");
        statusLabel_->setText("Configure a server to begin.");
        return;
    }
    serverLabel_->setText("Target: " + settings_.serverHost + ":" + settings_.serverPort);
    if (!configurationIsReady()) {
        securityLabel_->setText("Connection setup is incomplete or a credential file is unavailable. Review settings.");
        statusLabel_->setText("Connection settings need attention.");
        return;
    }
    securityLabel_->setText("Each request uses gRPC over mutual TLS. No persistent session is maintained.");
    statusLabel_->setText("Ready to submit a request; no persistent connection is open.");
}

bool MainWindow::configurationIsReady() const {
    bool portValid = false;
    const uint port = settings_.serverPort.toUInt(&portValid);
    if (settings_.serverHost.isEmpty() || !portValid || port == 0 || port > 65535) {
        return false;
    }
    const QStringList credentialPaths{
        settings_.serverCaFile,
        settings_.clientCertificateFile,
        settings_.clientPrivateKeyFile,
    };
    for (const QString& path : credentialPaths) {
        const QFileInfo info(path);
        if (!info.isFile() || !info.isReadable()) {
            return false;
        }
    }
    return true;
}

void MainWindow::appendActivity(const QString& message) {
    logsText_->append("[" + QDateTime::currentDateTime().toString("yyyy-MM-dd HH:mm:ss") +
                      "] " + message.toHtmlEscaped());
}

ClientSettings MainWindow::loadSettings() const {
    QSettings values;
    return {
        values.value("connection/serverHost").toString(),
        values.value("connection/serverPort", "50051").toString(),
        values.value("tls/serverCaFile").toString(),
        values.value("tls/clientCertificateFile").toString(),
        values.value("tls/clientPrivateKeyFile").toString(),
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
        updateConfigurationDisplay();
    }
    if (!configurationIsReady()) {
        updateConfigurationDisplay();
        QMessageBox::warning(this, "Connection setup needs attention",
                             "The server address, port, and readable CA/client credential files are required.");
        return;
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
        appendActivity("Error: tuna_sample_app was not found beside the GUI.");
        return;
    }

    requestTimedOut_ = false;
    requestCancelled_ = false;
    runWorkloadButton_->setEnabled(false);
    configureButton_->setEnabled(false);
    cancelWorkloadButton_->setEnabled(true);
    statusLabel_->setText("Sending request to the server...");
    resultLabel_->setText("Waiting for the remote result.");
    appendActivity("Submitting sample workload (" + QString::number(inputParts.size()) +
                   " values) to " + settings_.serverHost + ":" + settings_.serverPort + ".");
    workloadProcess_->setProgram(executable);
    workloadProcess_->setArguments(arguments);
    workloadProcess_->start();
    requestTimeout_->start(12000);
}

void MainWindow::onWorkloadFinished(int exitCode, int exitStatus) {
    requestTimeout_->stop();
    runWorkloadButton_->setEnabled(true);
    configureButton_->setEnabled(true);
    cancelWorkloadButton_->setEnabled(false);
    const QString standardOutput = QString::fromLocal8Bit(workloadProcess_->readAllStandardOutput());
    const QString standardError = QString::fromLocal8Bit(workloadProcess_->readAllStandardError());

    if (requestTimedOut_) {
        statusLabel_->setText("Request timed out");
        resultLabel_->setText("The remote request exceeded the client timeout.");
        appendActivity("Error: remote RPC exceeded the client timeout.");
        return;
    }
    if (requestCancelled_) {
        statusLabel_->setText("Request cancelled");
        resultLabel_->setText("The client process was cancelled. The server may have already completed the request.");
        appendActivity("Request cancelled by the user; remote completion is unknown.");
        return;
    }
    if (exitCode != 0 || exitStatus != static_cast<int>(QProcess::NormalExit)) {
        statusLabel_->setText("Request failed");
        resultLabel_->setText("The remote request failed. See Recent activity for the error.");
        const QString error = standardError.trimmed();
        appendActivity("Error: " + (error.isEmpty() ? "sample client exited unexpectedly." : error));
        return;
    }

    static const QRegularExpression resultPattern(R"(sum_of_squares=(\d+))");
    const auto match = resultPattern.match(standardOutput);
    if (!match.hasMatch()) {
        statusLabel_->setText("Invalid sample-client response");
        resultLabel_->setText("No valid result was returned. See Recent activity for details.");
        appendActivity("Error: sample client returned no recognizable result.");
        return;
    }

    const QString result = match.captured(1);
    statusLabel_->setText("Request succeeded");
    resultLabel_->setText("Remote result: " + result);
    appendActivity("Request succeeded; remote sum of squares = " + result + ".");
}
