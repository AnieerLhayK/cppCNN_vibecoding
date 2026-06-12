import QtQuick
import QtQuick.Controls

Rectangle {
    id: pill
    property bool positive: false
    property string text: ""

    implicitWidth: label.implicitWidth + 34
    implicitHeight: 30
    radius: 15
    color: positive ? "#15382F" : "#30243A"
    border.color: positive ? "#2D806A" : "#704A7C"

    Rectangle {
        width: 8
        height: 8
        radius: 4
        anchors.left: parent.left
        anchors.leftMargin: 12
        anchors.verticalCenter: parent.verticalCenter
        color: pill.positive ? "#45D3A4" : "#C985E7"
    }

    Label {
        id: label
        anchors.left: parent.left
        anchors.leftMargin: 25
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        text: pill.text
        color: pill.positive ? "#A8F0D8" : "#EAC7F5"
        font.pixelSize: 12
        font.weight: Font.DemiBold
        elide: Text.ElideRight
    }
}
