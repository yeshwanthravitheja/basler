import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 720
    title: "Basler"

    Timer {
        property var trigger_index: 0
        interval: 16
        running: true
        repeat: true
        onTriggered: {
            ++trigger_index
            image_display.source = `image://camera/${trigger_index}`
        }
    }

    RowLayout{
        spacing: 0
        width: window.width
        height: window.height

        Rectangle {
            id: image_display_container
            Layout.alignment: Qt.AlignCenter
            Layout.fillHeight: true
            Layout.fillWidth: true
            color: "#090909"

            Image {
                id: image_display
                anchors.fill: image_display_container
                fillMode: Image.PreserveAspectFit
                source: `image://camera/0`
            }
        }

        Rectangle {
            id: menu
            Layout.alignment: Qt.AlignCenter
            Layout.fillHeight: true
            Layout.minimumWidth: 320
            Layout.maximumWidth: 320
            color: "#191919"

            ScrollView {
                width: menu.width
                height: menu.height
                clip: true
                topPadding: 20
                leftPadding: 20
                rightPadding: 20
                bottomPadding: 20

                ColumnLayout {
                    width: menu.width - 40

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            text: "Width"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        SpinBox {
                            palette.button: "#393939"
                            palette.buttonText: "#FFFFFF"
                            palette.text: "#FFFFFF"
                            palette.base: "#191919"
                            palette.mid: "#494949"
                            palette.highlight: "#1E88E5"
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            from: 16
                            to: 656
                            value: 640
                            stepSize: 16
                            editable: true
                            enabled: configuration?.recording_name == null
                            font: monospace_font

                            onValueModified: {
                                configuration.width = value;
                            }
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            text: "Height"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        SpinBox {
                            palette.button: "#393939"
                            palette.buttonText: "#FFFFFF"
                            palette.text: "#FFFFFF"
                            palette.base: "#191919"
                            palette.mid: "#494949"
                            palette.highlight: "#1E88E5"
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            from: 1
                            to: 496
                            value: 480
                            stepSize: 1
                            editable: true
                            enabled: configuration?.recording_name == null
                            font: monospace_font

                            onValueModified: {
                                configuration.height = value;
                            }
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            text: "Maximum frame rate"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        SpinBox {
                            id: frame_rate_spinbox
                            palette.button: "#393939"
                            palette.buttonText: "#FFFFFF"
                            palette.text: "#FFFFFF"
                            palette.base: "#191919"
                            palette.mid: "#494949"
                            palette.highlight: "#1E88E5"
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            from: 1
                            to: 10000000
                            stepSize: 1
                            value: 5000
                            editable: true
                            enabled: configuration?.recording_name == null
                            font: monospace_font

                            validator: DoubleValidator {
                                bottom: frame_rate_spinbox.from
                                top: frame_rate_spinbox.to
                                decimals: 1
                                notation: DoubleValidator.StandardNotation
                            }

                            textFromValue: function(value, locale) {
                                return Number(value / 10).toLocaleString(locale, 'f', 1)
                            }

                            valueFromText: function(text, locale) {
                                return Math.round(Number.fromLocaleString(locale, text) * 10)
                            }

                            onValueModified: {
                                configuration.framerate = value;
                            }
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            text: "Exposure (µs)"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        SpinBox {
                            palette.button: "#393939"
                            palette.buttonText: "#FFFFFF"
                            palette.text: "#FFFFFF"
                            palette.base: "#191919"
                            palette.mid: "#494949"
                            palette.highlight: "#1E88E5"
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            from: 59
                            to: 10000000
                            value: 1000
                            stepSize: 1
                            editable: true
                            enabled: configuration?.recording_name == null
                            font: monospace_font

                            onValueModified: {
                                configuration.exposure = value;
                            }
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            text: "Gain"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        SpinBox {
                            id: gain_spinbox
                            palette.button: "#393939"
                            palette.buttonText: "#FFFFFF"
                            palette.text: "#FFFFFF"
                            palette.base: "#191919"
                            palette.mid: "#494949"
                            palette.highlight: "#1E88E5"
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignRight
                            from: 0
                            to: 1200
                            stepSize: 1
                            value: 0
                            editable: true
                            enabled: configuration?.recording_name == null
                            font: monospace_font

                            validator: DoubleValidator {
                                bottom: gain_spinbox.from
                                top: gain_spinbox.to
                                decimals: 1
                                notation: DoubleValidator.StandardNotation
                            }

                            textFromValue: function(value, locale) {
                                return Number(value / 100).toLocaleString(locale, 'f', 2)
                            }

                            valueFromText: function(text, locale) {
                                return Math.round(Number.fromLocaleString(locale, text) * 100)
                            }

                            onValueModified: {
                                configuration.gain = value;
                            }
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 10
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: "Calculated frame rate"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration && configuration.calculated_framerate ? configuration.calculated_framerate.toFixed(1) : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: "Measured frame rate"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration && configuration.measured_framerate ? configuration.measured_framerate.toFixed(1) : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: "USB buffer usage"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration ? `${configuration.queued_buffers} / ${configuration.maximum_queued_buffers}` : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 20

                        ComboBox {
                            model: ["Direct mode", "Circular buffer"]
                            enabled: configuration?.recording_name == null
                            onCurrentIndexChanged: {
                                configuration.mode = currentIndex;
                            }
                        }
                    }

                    RowLayout {
                        visible: configuration?.mode == 0
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: "RAM buffer"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration ? configuration.buffered_frames : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                    }

                    ColumnLayout {
                        visible: configuration?.mode == 1
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 5

                        RowLayout {
                            Layout.topMargin: 5
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            spacing: 20
                            Text {
                                Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                                text: "Duration"
                                color: "#CCCCCC"
                                font: monospace_font
                            }
                            ComboBox {
                                model: ["100 ms", "200 ms", "500 ms", "1 s", "2 s", "5 s", "10 s", "20 s"]
                                currentIndex: 5
                                enabled: configuration?.recording_name == null
                                onCurrentIndexChanged: {
                                    configuration.circular_buffer_duration = model[currentIndex];
                                }
                            }
                        }
                    }

                    RowLayout {
                        visible: configuration?.mode == 1
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 20
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: "Circular buffer"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration ? configuration.circular_buffer_usage : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                    }

                    RowLayout {
                        Layout.topMargin: 5
                        Layout.alignment: configuration?.recording_name == null ? Qt.AlignVCenter | Qt.AlignLeft : Qt.AlignVCenter | Qt.AlignRight
                        spacing: 20

                        Button {
                            property var click_index: 0
                            text: "Start recording"
                            visible: configuration?.recording_name == null
                            onClicked: {
                                ++click_index
                                configuration.start_recording = click_index
                            }
                        }

                        Button {
                            property var click_index: 0
                            text: "Stop recording"
                            visible: configuration?.recording_name != null && configuration?.mode == 0
                            onClicked: {
                                ++click_index
                                configuration.stop_recording = click_index
                            }
                        }
                    }

                    ColumnLayout {
                        visible: configuration?.recording_name != null
                        Layout.topMargin: 5
                        Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                        spacing: 5

                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: "Recording to"
                            color: "#CCCCCC"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration && configuration.recording_name ? configuration.recording_name : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration ? configuration.recording_duration : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration ? `${configuration.recording_frames} frames` : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                        Text {
                            Layout.alignment: Qt.AlignVCenter | Qt.AlignLeft
                            text: configuration ? configuration.recording_bytes : "-"
                            color: "#FFFFFF"
                            font: monospace_font
                        }
                    }
                }
            }
        }
    }
}
