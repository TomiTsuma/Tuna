#pragma once

#include <QString>
#include <QWidget>

struct ClientSettings {
    QString serverHost;
    QString serverPort;
    QString serverCaFile;
    QString clientCertificateFile;
    QString clientPrivateKeyFile;
};

bool openSettingsDialog(QWidget* parent, ClientSettings* settings);
