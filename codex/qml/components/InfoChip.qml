import QtQuick
import QtQuick.Controls

Rectangle {
    id: chip
    property string title: ""
    property string value: ""
    property bool complete: false

    implicitHeight: 42
    radius: 10
    color: complete ? "#132D2A" : "#172238"
    border.color: complete ? "#286D5D" : "#2B3B59"

    Row {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        Rectangle {
            anchors.verticalCenter: parent.verticalCenter
            width: 8
            height: 8
            radius: 4
            color: chip.complete ? "#45D3A4" : "#657492"
        }

        Column {
            anchors.verticalCenter: parent.verticalCenter
            width: parent.width - 18
            spacing: 1

            Label {
                width: parent.width
                text: chip.title
                color: "#7586A5"
                font.pixelSize: 10
                font.weight: Font.Bold
                font.letterSpacing: 0.7
                elide: Text.ElideRight
            }
            Label {
                width: parent.width
                text: chip.value
                color: chip.complete ? "#B7EBDD" : "#A8B4C9"
                font.pixelSize: 12
                font.weight: Font.DemiBold
                elide: Text.ElideRight
            }
        }
    }
}
