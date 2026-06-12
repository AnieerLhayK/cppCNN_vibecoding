import QtQuick
import QtQuick.Controls

ApplicationWindow {
    id: window
    width: 1200
    height: 760
    minimumWidth: 960
    minimumHeight: 640
    visible: true
    title: "cppCNN Traffic Sign Studio"
    color: "#0B1020"

    Rectangle {
        anchors.fill: parent
        color: window.color

        Column {
            anchors.centerIn: parent
            spacing: 12

            Label {
                anchors.horizontalCenter: parent.horizontalCenter
                text: "cppCNN Traffic Sign Studio"
                color: "#F4F7FF"
                font.pixelSize: 30
                font.weight: Font.DemiBold
            }

            Label {
                anchors.horizontalCenter: parent.horizontalCenter
                text: appController.statusText
                color: "#8FA2C9"
                font.pixelSize: 15
            }
        }
    }
}
