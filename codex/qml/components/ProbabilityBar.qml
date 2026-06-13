import QtQuick
import QtQuick.Controls

Item {
    id: root
    property int rank: 1
    property string label: ""
    property int classId: -1
    property real confidence: 0

    implicitHeight: 50

    Row {
        id: heading
        anchors.left: parent.left
        anchors.right: parent.right
        height: 22
        spacing: 8

        Label {
            width: 22
            text: root.rank
            color: root.rank === 1 ? "#AFA8FF" : "#7585A5"
            font.pixelSize: 13
            font.weight: Font.Bold
        }

        Label {
            width: Math.max(0, heading.width - 92)
            text: root.label
            color: "#E8ECF7"
            font.pixelSize: 13
            font.weight: root.rank === 1 ? Font.DemiBold : Font.Normal
            elide: Text.ElideRight
        }

        Label {
            width: 54
            text: (root.confidence * 100).toFixed(1) + "%"
            color: root.rank === 1 ? "#D5D1FF" : "#91A0BC"
            font.pixelSize: 13
            horizontalAlignment: Text.AlignRight
        }
    }

    Rectangle {
        id: track
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: heading.bottom
        anchors.topMargin: 8
        height: 8
        radius: 4
        color: "#202A40"

        Rectangle {
            height: parent.height
            radius: parent.radius
            width: Math.max(height, parent.width * Math.max(0, Math.min(1, root.confidence)))
            color: root.rank === 1 ? "#7168EB" : (root.rank === 2 ? "#4F72C9" : "#3C577F")

            Behavior on width {
                NumberAnimation { duration: 420; easing.type: Easing.OutCubic }
            }
        }
    }
}
