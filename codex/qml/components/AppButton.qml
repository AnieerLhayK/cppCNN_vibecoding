import QtQuick
import QtQuick.Controls

Button {
    id: control
    property bool accent: false
    property bool compact: false

    implicitHeight: compact ? 36 : 44
    implicitWidth: Math.max(compact ? 92 : 124, contentItem.implicitWidth + 28)
    hoverEnabled: true

    contentItem: Label {
        text: control.text
        color: control.enabled ? "#F7F9FF" : "#64728E"
        font.pixelSize: control.compact ? 13 : 14
        font.weight: Font.DemiBold
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
    }

    background: Rectangle {
        radius: 10
        color: {
            if (!control.enabled)
                return "#172033"
            if (control.down)
                return control.accent ? "#5146C9" : "#26334E"
            if (control.hovered)
                return control.accent ? "#7065EA" : "#202D47"
            return control.accent ? "#6257DD" : "#18243A"
        }
        border.color: control.accent ? "#8076F4" : "#2A3A58"

        Behavior on color {
            ColorAnimation { duration: 120 }
        }
    }
}
