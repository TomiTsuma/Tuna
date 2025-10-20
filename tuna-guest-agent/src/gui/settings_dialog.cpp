// Minimal Settings dialog implementation (UI-only)
#include <QDialog>
#include <QFormLayout>
#include <QLineEdit>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QCheckBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QPushButton>

class SettingsDialog : public QDialog {
    Q_OBJECT
public:
    explicit SettingsDialog(QWidget* parent = nullptr) : QDialog(parent) {
        setWindowTitle("Settings");
        auto* form = new QFormLayout(this);

        serverEdit_ = new QLineEdit(this);
        serverEdit_->setPlaceholderText("server.region.tuna.cloud");
        portEdit_ = new QLineEdit(this);
        portEdit_->setPlaceholderText("443");
        protocolBox_ = new QComboBox(this);
        protocolBox_->addItems({"QUIC/HTTP3", "gRPC/TLS"});
        mtlsCheck_ = new QCheckBox("Enable mutual TLS (client certificate)", this);

        auto* certRow = new QWidget(this);
        auto* certLay = new QHBoxLayout(certRow);
        certLay->setContentsMargins(0,0,0,0);
        certEdit_ = new QLineEdit(this);
        auto* browseBtn = new QPushButton("Browse", this);
        connect(browseBtn, &QPushButton::clicked, this, [this]() {
            auto path = QFileDialog::getOpenFileName(this, "Select certificate", {}, "Certificates (*.pem *.crt)");
            if (!path.isEmpty()) certEdit_->setText(path);
        });
        certLay->addWidget(certEdit_);
        certLay->addWidget(browseBtn);

        form->addRow("Server", serverEdit_);
        form->addRow("Port", portEdit_);
        form->addRow("Protocol", protocolBox_);
        form->addRow(mtlsCheck_);
        form->addRow("Client certificate", certRow);

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        form->addRow(buttons);

        resize(520, 260);
    }

private:
    QLineEdit* serverEdit_{};
    QLineEdit* portEdit_{};
    QComboBox* protocolBox_{};
    QCheckBox* mtlsCheck_{};
    QLineEdit* certEdit_{};
};

static void openSettingsDialogInternal(QWidget* parent) {
    SettingsDialog dlg(parent);
    dlg.exec();
}

void openSettingsDialog(QWidget* parent) {
    openSettingsDialogInternal(parent);
}

// Ensure Qt's meta-object code is generated for this cpp-defined QObject
#include "settings_dialog.moc"


