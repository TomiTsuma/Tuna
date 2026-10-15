#include "settings_dialog.h"

#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QList>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>

namespace {

class SettingsDialog final : public QDialog {
public:
    SettingsDialog(const ClientSettings& settings, QWidget* parent)
        : QDialog(parent) {
        setWindowTitle("Tuna Client Settings");
        auto* form = new QFormLayout(this);

        hostEdit_ = new QLineEdit(settings.serverHost, this);
        hostEdit_->setPlaceholderText("tuna.example.net");
        hostEdit_->setToolTip("Enter a DNS hostname or IPv4 address present in the server certificate.");

        portEdit_ = new QLineEdit(settings.serverPort, this);
        portEdit_->setPlaceholderText("50051");
        portEdit_->setValidator(new QIntValidator(1, 65535, portEdit_));

        caEdit_ = addFilePicker("Server CA certificate", settings.serverCaFile,
                                "Select the CA certificate used to validate the server.",
                                "CA certificates (*.pem *.crt)");
        clientCertificateEdit_ = addFilePicker(
            "Client certificate", settings.clientCertificateFile,
            "Select the administrator-provisioned client certificate.",
            "Client certificates (*.pem *.crt)");
        clientPrivateKeyEdit_ = addFilePicker(
            "Client private key", settings.clientPrivateKeyFile,
            "Select the matching client private key. Its file permissions must be restricted.",
            "Private keys (*.pem *.key)");

        form->addRow("Server host", hostEdit_);
        form->addRow("Server port", portEdit_);
        form->addRow("Transport", new QLabel("gRPC over TLS; mutual TLS is required", this));
        form->addRow("Server CA", caEdit_->parentWidget());
        form->addRow("Client certificate", clientCertificateEdit_->parentWidget());
        form->addRow("Client private key", clientPrivateKeyEdit_->parentWidget());

        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, this);
        connect(buttons, &QDialogButtonBox::accepted, this, [this]() { validateAndAccept(); });
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        form->addRow(buttons);

        resize(600, 280);
    }

    ClientSettings settings() const {
        return {
            hostEdit_->text().trimmed(),
            portEdit_->text().trimmed(),
            caEdit_->text().trimmed(),
            clientCertificateEdit_->text().trimmed(),
            clientPrivateKeyEdit_->text().trimmed(),
        };
    }

private:
    QLineEdit* addFilePicker(const QString& label, const QString& path,
                             const QString& help, const QString& filter) {
        auto* row = new QWidget(this);
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        auto* edit = new QLineEdit(path, row);
        edit->setToolTip(help);
        auto* browse = new QPushButton("Browse...", row);
        connect(browse, &QPushButton::clicked, this, [this, edit, label, filter]() {
            const QString selected = QFileDialog::getOpenFileName(this, label, edit->text(), filter);
            if (!selected.isEmpty()) {
                edit->setText(selected);
            }
        });
        layout->addWidget(edit);
        layout->addWidget(browse);
        return edit;
    }

    bool readableFile(const QString& path) const {
        const QFileInfo info(path);
        if (!info.exists() || !info.isFile()) {
            return false;
        }
        QFile file(path);
        return file.open(QIODevice::ReadOnly);
    }

    void validateAndAccept() {
        static const QRegularExpression hostPattern(
            R"(^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)*$)");
        if (!hostPattern.match(hostEdit_->text().trimmed()).hasMatch()) {
            QMessageBox::warning(this, "Invalid server host",
                                 "Enter a DNS hostname or IPv4 address without a scheme or path.");
            return;
        }
        if (!portEdit_->hasAcceptableInput() || portEdit_->text().isEmpty()) {
            QMessageBox::warning(this, "Invalid server port", "Enter a port from 1 to 65535.");
            return;
        }
        if (!readableFile(caEdit_->text()) ||
            !readableFile(clientCertificateEdit_->text()) ||
            !readableFile(clientPrivateKeyEdit_->text())) {
            QMessageBox::warning(this, "Invalid TLS credentials",
                                 "The CA, client certificate, and private key must be readable files.");
            return;
        }
        accept();
    }

    QLineEdit* hostEdit_{};
    QLineEdit* portEdit_{};
    QLineEdit* caEdit_{};
    QLineEdit* clientCertificateEdit_{};
    QLineEdit* clientPrivateKeyEdit_{};
};

}  // namespace

bool openSettingsDialog(QWidget* parent, ClientSettings* settings) {
    if (settings == nullptr) {
        return false;
    }
    SettingsDialog dialog(*settings, parent);
    if (dialog.exec() != QDialog::Accepted) {
        return false;
    }
    *settings = dialog.settings();
    return true;
}
