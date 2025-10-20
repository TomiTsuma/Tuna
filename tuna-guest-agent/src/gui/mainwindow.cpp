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
#include <QTimer>
#include <QTime>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QSystemTrayIcon>
#include <QIcon>
#include <QTextEdit>
#include <QTableWidget>
#include <QHeaderView>
#include <QRandomGenerator>

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
    auto* tabs = new QTabWidget(this);
    tabs->addTab(buildDashboard(), "Dashboard");
    tabs->addTab(buildPerformance(), "Performance");
    tabs->addTab(buildProcesses(), "Processes");
    tabs->addTab(buildTunnel(), "Tunnel");
    tabs->addTab(buildLogs(), "Logs");
    setCentralWidget(tabs);
    resize(1180, 760);

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

    statusLabel_ = makeTitle("Connected to Tuna Server");
    statusLabel_->setProperty("accent", true);
    serverLabel_ = new QLabel("server-us-west-2.tuna.cloud");
    uptimeLabel_ = new QLabel("Uptime\n2h 34m");
    tlsLabel_ = new QLabel("Encryption\nTLS 1.3");
    protocolLabel_ = new QLabel("Protocol\nQUIC/HTTP3");
    disconnectButton_ = new QPushButton("Disconnect");
    disconnectButton_->setMinimumWidth(140);
    connect(disconnectButton_, &QPushButton::clicked, this, &MainWindow::onToggleConnection);

    headerLayout->addWidget(statusLabel_, 0, 0, 1, 3);
    headerLayout->addWidget(serverLabel_, 1, 0, 1, 3);
    headerLayout->addWidget(uptimeLabel_, 2, 0);
    headerLayout->addWidget(tlsLabel_, 2, 1);
    headerLayout->addWidget(protocolLabel_, 2, 2);
    headerLayout->addWidget(disconnectButton_, 1, 4, 2, 1);

    v->addWidget(card(header));

    // Metrics row
    auto* row = new QHBoxLayout();
    row->setSpacing(16);

    auto* cpuBox = new QWidget(root);
    auto* cpuLay = new QVBoxLayout(cpuBox);
    cpuLay->addWidget(new QLabel("CPU Offload"));
    cpuBar_ = new QProgressBar(cpuBox);
    cpuBar_->setRange(0, 100);
    cpuBar_->setValue(87);
    cpuLay->addWidget(cpuBar_);

    auto* gpuBox = new QWidget(root);
    auto* gpuLay = new QVBoxLayout(gpuBox);
    gpuLay->addWidget(new QLabel("GPU Usage"));
    gpuBar_ = new QProgressBar(gpuBox);
    gpuBar_->setRange(0, 100);
    gpuBar_->setValue(92);
    gpuLay->addWidget(gpuBar_);

    auto* latBox = new QWidget(root);
    auto* latLay = new QVBoxLayout(latBox);
    latLay->addWidget(new QLabel("Tunnel Latency"));
    latencyBar_ = new QProgressBar(latBox);
    latencyBar_->setRange(0, 200);
    latencyBar_->setValue(12);
    latLay->addWidget(latencyBar_);

    row->addWidget(card(cpuBox));
    row->addWidget(card(gpuBox));
    row->addWidget(card(latBox));

    auto* rowWrap = new QWidget(root);
    rowWrap->setLayout(row);
    v->addWidget(rowWrap);

    // Simulated updates
    auto* timer = new QTimer(root);
    connect(timer, &QTimer::timeout, this, [this]() {
        cpuBar_->setValue((cpuBar_->value() + 3) % 100);
        gpuBar_->setValue((gpuBar_->value() + 5) % 100);
        int next = (latencyBar_->value() + 7) % 200;
        latencyBar_->setValue(next);
    });
    timer->start(3000);

    return root;
}

QWidget* MainWindow::buildPerformance() {
    auto* w = new QWidget(this);
    auto* l = new QVBoxLayout(w);
    auto* title = makeTitle("Live Metrics");
    l->addWidget(title);
    auto* table = new QTableWidget(0, 3, w);
    table->setHorizontalHeaderLabels({"Metric", "Current", "Trend"});
    table->horizontalHeader()->setStretchLastSection(true);
    l->addWidget(table);

    auto addRow = [table](const QString& name) {
        int r = table->rowCount();
        table->insertRow(r);
        table->setItem(r, 0, new QTableWidgetItem(name));
        table->setItem(r, 1, new QTableWidgetItem("0"));
        table->setItem(r, 2, new QTableWidgetItem("↗"));
    };
    addRow("CPU Offload %");
    addRow("GPU Util %");
    addRow("Tunnel RTT ms");
    addRow("Throughput MB/s");

    auto* timer = new QTimer(w);
    connect(timer, &QTimer::timeout, this, [table]() {
        for (int r = 0; r < table->rowCount(); ++r) {
            int val = QRandomGenerator::global()->bounded(1, 100);
            table->item(r, 1)->setText(QString::number(val));
            table->item(r, 2)->setText(val % 2 ? "↗" : "↘");
        }
    });
    timer->start(2000);
    return w;
}

QWidget* MainWindow::buildProcesses() {
    auto* w = new QWidget(this);
    auto* l = new QVBoxLayout(w);
    l->addWidget(makeTitle("Hooked Processes"));
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
    addProc("Photoshop.exe", 1234, "D3D11, FileIO", "Active");
    addProc("Blender.exe", 5678, "CUDA, FileIO", "Active");
    addProc("python.exe", 9012, "OpenCL, FileIO", "Idle");
    return w;
}

QWidget* MainWindow::buildTunnel() {
    auto* w = new QWidget(this);
    auto* l = new QVBoxLayout(w);
    l->addWidget(makeTitle("Tunnel Diagnostics"));
    auto* grid = new QGridLayout();
    l->addLayout(grid);

    grid->addWidget(new QLabel("Protocol:"), 0, 0);
    grid->addWidget(new QLabel("QUIC/HTTP3"), 0, 1);
    grid->addWidget(new QLabel("Cipher:"), 1, 0);
    grid->addWidget(new QLabel("TLS 1.3 AES-256-GCM"), 1, 1);
    grid->addWidget(new QLabel("Heartbeat:"), 2, 0);
    grid->addWidget(new QLabel("OK (3s)"), 2, 1);
    grid->addWidget(new QLabel("Session ID:"), 3, 0);
    grid->addWidget(new QLabel("abcd-1234"), 3, 1);

    auto* btnRow = new QHBoxLayout();
    auto* reconnect = new QPushButton("Reconnect");
    auto* safeMode = new QPushButton("Enable Safe Mode");
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
    auto* text = new QTextEdit(w);
    text->setReadOnly(true);
    l->addWidget(text);

    auto* timer = new QTimer(w);
    connect(timer, &QTimer::timeout, this, [text]() {
        static int n = 0;
        text->append(QString("[%1] Tunnel heartbeat OK, RTT=%2ms")
                     .arg(QTime::currentTime().toString())
                     .arg(QRandomGenerator::global()->bounded(5, 60)));
        if (++n % 5 == 0) {
            text->append("Info: Hooks healthy; GPU path zero-copy active");
        }
    });
    timer->start(1500);
    return w;
}

void MainWindow::buildMenus() {
    auto* file = menuBar()->addMenu("File");
    auto* actSettings = file->addAction("Settings...");
    auto* actExit = file->addAction("Exit");
    auto* conn = menuBar()->addMenu("Connection");
    auto* actToggle = conn->addAction("Disconnect");
    auto* view = menuBar()->addMenu("View");
    auto* actLogs = view->addAction("Logs");

    connect(actExit, &QAction::triggered, this, [this]() { close(); });
    connect(actToggle, &QAction::triggered, this, [this, actToggle]() {
        onToggleConnection();
        actToggle->setText(connected_ ? "Disconnect" : "Connect");
    });
    connect(actLogs, &QAction::triggered, this, [this]() {
        // Switch to Logs tab
        auto* tabs = findChild<QTabWidget*>();
        if (tabs) tabs->setCurrentIndex(4);
    });

    connect(actSettings, &QAction::triggered, this, [this]() {
        // Lazy include to avoid header coupling
        extern void openSettingsDialog(QWidget* parent);
        openSettingsDialog(this);
    });
}

void MainWindow::buildTray() {
    tray_ = new QSystemTrayIcon(QIcon(), this);
    tray_->setToolTip("Tuna Guest Agent");
    auto* menu = new QMenu(this);
    auto* actOpen = menu->addAction("Open");
    auto* actToggle = menu->addAction("Disconnect");
    auto* actQuit = menu->addAction("Exit");
    connect(actOpen, &QAction::triggered, this, [this]() { showNormal(); activateWindow(); });
    connect(actToggle, &QAction::triggered, this, [this, actToggle]() {
        onToggleConnection();
        actToggle->setText(connected_ ? "Disconnect" : "Connect");
    });
    connect(actQuit, &QAction::triggered, this, [this]() { close(); });
    tray_->setContextMenu(menu);
    tray_->show();
}

void MainWindow::onToggleConnection() {
    connected_ = !connected_;
    if (connected_) {
        statusLabel_->setText("Connected to Tuna Server");
        disconnectButton_->setText("Disconnect");
    } else {
        statusLabel_->setText("Disconnected");
        disconnectButton_->setText("Connect");
    }
}


