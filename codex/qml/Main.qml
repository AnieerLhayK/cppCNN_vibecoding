import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts
import CppCNNGui

ApplicationWindow {
    id: window
    width: 1280
    height: 820
    minimumWidth: 1100
    minimumHeight: 700
    visible: true
    title: "cppCNN Traffic Sign Studio"
    color: "#0A0F1D"

    readonly property color textPrimary: "#F3F6FD"
    readonly property color textSecondary: "#92A1BC"
    readonly property color accent: "#7168EB"

    Shortcut {
        sequence: StandardKey.Open
        onActivated: imageDialog.open()
    }
    Shortcut {
        sequence: "Ctrl+M"
        onActivated: modelDialog.open()
    }
    Shortcut {
        sequence: "Ctrl+Return"
        enabled: appController.modelLoaded && appController.imageLoaded && !appController.busy
        onActivated: appController.predict()
    }
    Shortcut {
        sequence: "Escape"
        enabled: appController.imageLoaded && !appController.busy
        onActivated: appController.clearImage()
    }

    FileDialog {
        id: imageDialog
        title: "Choose a traffic sign image"
        nameFilters: ["Images (*.ppm *.pnm *.png *.jpg *.jpeg *.bmp)", "All files (*)"]
        onAccepted: appController.loadImage(selectedFile)
    }

    FileDialog {
        id: modelDialog
        title: "Choose a cppCNN model"
        nameFilters: ["cppCNN model (*.bin)", "All files (*)"]
        onAccepted: appController.loadModel(selectedFile)
    }

    Dialog {
        id: aboutDialog
        title: "About cppCNN Traffic Sign Studio"
        modal: true
        anchors.centerIn: parent
        width: 480
        standardButtons: Dialog.Ok

        contentItem: Label {
            padding: 18
            text: "cppCNN Traffic Sign Studio " + appController.applicationVersion
                + "\n\nA pure C++ LeNet-style CNN demonstration for GTSRB traffic-sign recognition."
                + "\n\nThe Tensor, convolution, pooling, activation, fully connected, softmax, loss, training, persistence, and inference code is implemented in this repository. Qt is used for the desktop interface and image decoding."
            color: window.textPrimary
            wrapMode: Text.WordWrap
            lineHeight: 1.3
        }

        background: Rectangle {
            color: "#151E31"
            radius: 14
            border.color: "#2B3A58"
        }
    }

    Dialog {
        id: settingsDialog
        title: "Runtime details"
        modal: true
        anchors.centerIn: parent
        width: 650
        standardButtons: Dialog.Close

        contentItem: ColumnLayout {
            spacing: 12

            Label {
                text: "Application " + appController.applicationVersion + "\n" + appController.modelDetails
                color: window.textPrimary
                font.pixelSize: 14
                wrapMode: Text.WordWrap
                Layout.fillWidth: true
            }
            Label {
                text: "Model\n" + (appController.modelPath || "Not loaded")
                color: window.textSecondary
                font.pixelSize: 13
                wrapMode: Text.WrapAnywhere
                Layout.fillWidth: true
            }
            Label {
                text: "Labels\n" + (appController.labelsPath || "Numeric fallback labels")
                color: window.textSecondary
                font.pixelSize: 13
                wrapMode: Text.WrapAnywhere
                Layout.fillWidth: true
            }
        }

        background: Rectangle {
            color: "#151E31"
            radius: 14
            border.color: "#2B3A58"
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 72
            color: "#0D1424"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 26
                anchors.rightMargin: 26
                spacing: 14

                ColumnLayout {
                    spacing: 1
                    Layout.fillWidth: true

                    Label {
                        text: "cppCNN Traffic Sign Studio"
                        color: window.textPrimary
                        font.pixelSize: 21
                        font.weight: Font.DemiBold
                    }
                    Label {
                        text: "Pure C++ convolutional neural network"
                        color: "#71809C"
                        font.pixelSize: 12
                    }
                }

                StatusPill {
                    positive: appController.modelLoaded
                    text: appController.modelStatus
                }

                Rectangle {
                    implicitWidth: versionLabel.implicitWidth + 22
                    implicitHeight: 30
                    radius: 15
                    color: "#1C1B36"
                    border.color: "#46417A"

                    Label {
                        id: versionLabel
                        anchors.centerIn: parent
                        text: "v" + appController.applicationVersion
                        color: "#BDB7F8"
                        font.pixelSize: 12
                        font.weight: Font.DemiBold
                    }
                }

                AppButton {
                    text: "Settings"
                    compact: true
                    toolTip: "View model paths and runtime details"
                    onClicked: settingsDialog.open()
                }
                AppButton {
                    text: "About"
                    compact: true
                    toolTip: "About this pure C++ CNN project"
                    onClicked: aboutDialog.open()
                }
            }

            Rectangle {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                height: 1
                color: "#1D2940"
            }
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 22
                spacing: 16

                Rectangle {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 50
                    radius: 12
                    color: appController.modelLoaded ? "#151E2D" : "#2B2232"
                    border.color: appController.modelLoaded ? "#26334D" : "#6A485F"

                    RowLayout {
                        anchors.fill: parent
                        anchors.leftMargin: 14
                        anchors.rightMargin: 10
                        spacing: 8

                        Label {
                            text: "Model"
                            color: "#71809C"
                            font.pixelSize: 11
                            font.weight: Font.Bold
                            font.letterSpacing: 0.8
                        }

                        ComboBox {
                            id: modelCombo
                            Layout.preferredWidth: 300
                            Layout.maximumWidth: 360
                            leftPadding: 10
                            model: appController.availableModels
                            textRole: "name"
                            valueRole: "path"
                            currentIndex: appController.currentModelIndex >= 0 ? appController.currentModelIndex : 0
                            enabled: appController.availableModels.length > 0 && !appController.busy

                            onActivated: function(index) {
                                const path = modelCombo.currentValue
                                if (path)
                                    appController.selectModelByPath(path)
                            }

                            delegate: ItemDelegate {
                                width: modelCombo.width
                                leftPadding: 10
                                contentItem: Column {
                                    spacing: 1
                                    Label {
                                        text: modelData.name || ""
                                        color: "#F3F6FD"
                                        font.pixelSize: 13
                                    }
                                    Label {
                                        visible: modelData.classCount > 0
                                        text: (modelData.classCount || "?") + " classes  ·  " + (modelData.architecture || "?")
                                        color: "#92A1BC"
                                        font.pixelSize: 10
                                    }
                                }
                                background: Rectangle {
                                    color: modelCombo.highlightedIndex === index ? "#1A253F" : "transparent"
                                }
                            }

                            contentItem: RowLayout {
                                spacing: 6
                                Label {
                                    text: modelCombo.currentText || "No .bin models found"
                                    color: modelCombo.currentText ? "#F3F6FD" : "#6F7F9B"
                                    font.pixelSize: 13
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                            }

                            background: Rectangle {
                                radius: 8
                                color: "#0B1120"
                                border.color: "#26334D"
                                border.width: 1
                            }

                            indicator: Rectangle {
                                x: modelCombo.width - width - 8
                                y: (modelCombo.height - height) / 2
                                width: 20
                                height: 20
                                color: "transparent"
                                Label {
                                    anchors.centerIn: parent
                                    text: "▾"
                                    color: "#92A1BC"
                                    font.pixelSize: 12
                                }
                            }
                        }

                        AppButton {
                            text: "Browse"
                            compact: true
                            toolTip: "Choose a model file (Ctrl+M)"
                            onClicked: modelDialog.open()
                        }

                        Item { Layout.fillWidth: true }

                        Rectangle {
                            visible: appController.modelLoaded
                            implicitWidth: classBadge.implicitWidth + 18
                            implicitHeight: 24
                            radius: 12
                            color: "#172238"
                            border.color: "#2A3B5D"
                            Label {
                                id: classBadge
                                anchors.centerIn: parent
                                text: appController.classCount + " classes"
                                color: "#A9B6CF"
                                font.pixelSize: 11
                            }
                        }

                        Label {
                            visible: !appController.modelLoaded
                            text: "No model — recognition disabled"
                            color: "#F0CADC"
                            font.pixelSize: 12
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 16

                    Panel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        Layout.preferredWidth: 760

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 12

                            RowLayout {
                                Layout.fillWidth: true

                                ColumnLayout {
                                    spacing: 2
                                    Layout.fillWidth: true
                                    Label {
                                        text: "Image inspection"
                                        color: window.textPrimary
                                        font.pixelSize: 17
                                        font.weight: Font.DemiBold
                                    }
                                    Label {
                                        text: appController.imageLoaded ? appController.imagePath : "PPM, PNG, JPEG, or BMP"
                                        color: window.textSecondary
                                        font.pixelSize: 12
                                        elide: Text.ElideMiddle
                                        Layout.maximumWidth: 510
                                    }
                                    Label {
                                        visible: appController.imageLoaded
                                        text: appController.imageDetails + " | CNN input 32 x 32 RGB"
                                        color: "#687A9B"
                                        font.pixelSize: 11
                                    }
                                }

                                AppButton {
                                    text: "Clear"
                                    compact: true
                                    toolTip: "Clear image (Esc)"
                                    enabled: appController.imageLoaded && !appController.busy
                                    onClicked: appController.clearImage()
                                }
                                AppButton {
                                    text: "Open image"
                                    compact: true
                                    accent: true
                                    toolTip: "Open image (Ctrl+O)"
                                    enabled: !appController.busy
                                    onClicked: imageDialog.open()
                                }
                            }

                            Rectangle {
                                id: imageCanvas
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                Layout.minimumHeight: 300
                                radius: 12
                                color: "#0B1120"
                                border.color: dropArea.containsDrag ? window.accent : "#26334D"
                                border.width: dropArea.containsDrag ? 2 : 1
                                clip: true

                                Image {
                                    id: preview
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    source: appController.imageUrl
                                    visible: appController.imageLoaded
                                    fillMode: Image.PreserveAspectFit
                                    asynchronous: true
                                    cache: false
                                }

                                Column {
                                    anchors.centerIn: parent
                                    width: Math.min(parent.width - 60, 430)
                                    spacing: 12
                                    visible: !appController.imageLoaded

                                    Rectangle {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        width: 62
                                        height: 62
                                        radius: 16
                                        color: "#19243A"
                                        border.color: "#34486D"

                                        Label {
                                            anchors.centerIn: parent
                                            text: "+"
                                            color: "#AFA8FF"
                                            font.pixelSize: 34
                                            font.weight: Font.Light
                                        }
                                    }
                                    Label {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        text: dropArea.containsDrag ? "Release to load image" : "Drop a traffic sign image here"
                                        color: window.textPrimary
                                        font.pixelSize: 18
                                        font.weight: Font.DemiBold
                                    }
                                    Label {
                                        anchors.horizontalCenter: parent.horizontalCenter
                                        width: parent.width
                                        text: "Or use Open image to choose a local file. It will be resized to 32 x 32 and normalized for the CNN."
                                        color: window.textSecondary
                                        font.pixelSize: 13
                                        horizontalAlignment: Text.AlignHCenter
                                        wrapMode: Text.WordWrap
                                    }
                                }

                                DropArea {
                                    id: dropArea
                                    anchors.fill: parent
                                    enabled: !appController.busy
                                    onDropped: function(drop) {
                                        if (drop.hasUrls && drop.urls.length > 0) {
                                            appController.loadImage(drop.urls[0])
                                            drop.acceptProposedAction()
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Panel {
                        Layout.fillHeight: true
                        Layout.preferredWidth: 380
                        Layout.minimumWidth: 340

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 20
                            spacing: 12

                            Label {
                                text: "Prediction"
                                color: window.textPrimary
                                font.pixelSize: 17
                                font.weight: Font.DemiBold
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                InfoChip {
                                    Layout.fillWidth: true
                                    title: "MODEL"
                                    value: appController.modelLoaded ? "Ready" : "Missing"
                                    complete: appController.modelLoaded
                                }
                                InfoChip {
                                    Layout.fillWidth: true
                                    title: "IMAGE"
                                    value: appController.imageLoaded ? "Loaded" : "Waiting"
                                    complete: appController.imageLoaded
                                }
                                InfoChip {
                                    Layout.fillWidth: true
                                    title: "RESULT"
                                    value: appController.busy
                                        ? "Running"
                                        : (appController.confidence > 0 ? "Complete" : "Waiting")
                                    complete: appController.confidence > 0
                                }
                            }

                            Rectangle {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 145
                                radius: 13
                                color: "#10192B"
                                border.color: appController.confidence > 0 ? "#4C477F" : "#25344F"

                                Column {
                                    anchors.fill: parent
                                    anchors.margins: 18
                                    spacing: 8

                                    Label {
                                        text: appController.busy ? "RUNNING CNN" : "TOP-1 RESULT"
                                        color: appController.busy ? "#B9B3FF" : "#7183A3"
                                        font.pixelSize: 11
                                        font.weight: Font.Bold
                                        font.letterSpacing: 1.2
                                    }
                                    Label {
                                        width: parent.width
                                        text: appController.predictionLabel
                                        color: window.textPrimary
                                        font.pixelSize: 23
                                        font.weight: Font.DemiBold
                                        wrapMode: Text.WordWrap
                                        maximumLineCount: 2
                                        elide: Text.ElideRight
                                    }
                                    Row {
                                        spacing: 8
                                        visible: appController.confidence > 0
                                        Label {
                                            text: (appController.confidence * 100).toFixed(1) + "%"
                                            color: "#B9B3FF"
                                            font.pixelSize: 25
                                            font.weight: Font.Bold
                                        }
                                        Label {
                                            anchors.baseline: parent.children[0].baseline
                                            text: "confidence"
                                            color: window.textSecondary
                                            font.pixelSize: 12
                                        }
                                    }
                                    Label {
                                        visible: appController.inferenceMilliseconds > 0
                                        text: "Inference " + appController.inferenceMilliseconds + " ms"
                                        color: "#7F91B2"
                                        font.pixelSize: 12
                                    }
                                }

                                BusyIndicator {
                                    anchors.right: parent.right
                                    anchors.bottom: parent.bottom
                                    anchors.margins: 16
                                    running: appController.busy
                                    visible: running
                                    palette.highlight: window.accent
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                spacing: 4

                                RowLayout {
                                    Layout.fillWidth: true
                                    Label {
                                        text: "Top probabilities"
                                        color: "#CBD3E3"
                                        font.pixelSize: 13
                                        font.weight: Font.DemiBold
                                        Layout.fillWidth: true
                                    }
                                    Label {
                                        text: appController.topResults.length > 0 ? "Top 3" : "No result"
                                        color: "#6F809F"
                                        font.pixelSize: 11
                                    }
                                }

                                Repeater {
                                    model: appController.topResults
                                    ProbabilityBar {
                                        Layout.fillWidth: true
                                        rank: index + 1
                                        label: modelData.label
                                        classId: modelData.classId
                                        confidence: modelData.confidence
                                    }
                                }

                                Item {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    visible: appController.topResults.length === 0

                                    Label {
                                        anchors.centerIn: parent
                                        width: parent.width - 20
                                        text: appController.modelLoaded
                                            ? "Load an image and run recognition to inspect class probabilities."
                                            : "Select a trained model to enable recognition."
                                        color: "#6F7F9B"
                                        font.pixelSize: 13
                                        horizontalAlignment: Text.AlignHCenter
                                        wrapMode: Text.WordWrap
                                    }
                                }
                            }

                            AppButton {
                                Layout.fillWidth: true
                                text: appController.busy ? "Recognizing..." : "Recognize sign"
                                accent: true
                                toolTip: "Run CNN inference (Ctrl+Enter)"
                                enabled: appController.modelLoaded && appController.imageLoaded && !appController.busy
                                onClicked: appController.predict()
                            }
                        }
                    }
                }

                Panel {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 132

                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 15
                        spacing: 10

                        RowLayout {
                            Layout.fillWidth: true
                            Label {
                                text: "Class showcase"
                                color: window.textPrimary
                                font.pixelSize: 15
                                font.weight: Font.DemiBold
                                Layout.fillWidth: true
                            }
                            Label {
                                text: "5 distinct classes  |  Click to recognize"
                                color: window.textSecondary
                                font.pixelSize: 11
                            }
                        }

                        RowLayout {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            spacing: 10

                            Repeater {
                                model: appController.demoImages

                                Rectangle {
                                    Layout.fillWidth: true
                                    Layout.fillHeight: true
                                    Layout.minimumWidth: 130
                                    radius: 10
                                    color: sampleMouse.containsMouse ? "#202D47" : "#172238"
                                    border.color: appController.imageUrl.toString() === modelData.url.toString()
                                        ? "#8B82FF"
                                        : (sampleMouse.containsMouse ? "#6257DD" : "#2A3955")
                                    border.width: appController.imageUrl.toString() === modelData.url.toString() ? 2 : 1

                                    RowLayout {
                                        anchors.fill: parent
                                        anchors.margins: 8
                                        spacing: 9

                                        Rectangle {
                                            Layout.preferredWidth: 54
                                            Layout.fillHeight: true
                                            radius: 8
                                            color: "#0B1120"
                                            clip: true

                                            Image {
                                                anchors.fill: parent
                                                anchors.margins: 4
                                                source: modelData.url
                                                fillMode: Image.PreserveAspectFit
                                                asynchronous: true
                                            }
                                        }
                                        Label {
                                            Layout.fillWidth: true
                                            text: modelData.name
                                            color: "#C8D1E3"
                                            font.pixelSize: 11
                                            wrapMode: Text.WordWrap
                                            maximumLineCount: 3
                                            elide: Text.ElideRight
                                        }
                                    }

                                    MouseArea {
                                        id: sampleMouse
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        enabled: !appController.busy
                                        cursorShape: Qt.PointingHandCursor
                                        onClicked: {
                                            appController.loadImage(modelData.url)
                                            if (appController.modelLoaded)
                                                appController.predict()
                                        }
                                    }

                                    Behavior on color {
                                        ColorAnimation { duration: 120 }
                                    }
                                }
                            }

                            Label {
                                visible: appController.demoImages.length === 0
                                Layout.fillWidth: true
                                text: "Demo images were not found beside the application."
                                color: window.textSecondary
                                horizontalAlignment: Text.AlignHCenter
                            }
                        }
                    }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: appController.errorText ? 48 : 36
            color: appController.errorText ? "#281923" : "#0D1424"

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 22
                anchors.rightMargin: 22
                spacing: 14

                Rectangle {
                    width: 7
                    height: 7
                    radius: 4
                    color: appController.errorText ? "#EE7B9C" : (appController.busy ? "#AFA8FF" : "#45D3A4")
                }
                Label {
                    text: appController.errorText || appController.statusText
                    color: appController.errorText ? "#F4B5C7" : "#8292AF"
                    font.pixelSize: 12
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
                Label {
                    visible: appController.modelLoaded
                    text: appController.modelPath
                    color: "#596985"
                    font.pixelSize: 11
                    elide: Text.ElideMiddle
                    Layout.maximumWidth: 390
                }
                Label {
                    text: "Ctrl+O Open  |  Ctrl+M Model  |  Ctrl+Enter Recognize  |  Esc Clear"
                    color: "#4F607C"
                    font.pixelSize: 10
                    visible: window.width >= 1250
                }
            }
        }
    }
}
