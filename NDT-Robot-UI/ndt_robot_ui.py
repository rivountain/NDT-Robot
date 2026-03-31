import asyncio
from enum import Enum
from cobs import cobs
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSpinBox,
    QLabel,
    QPushButton,
)
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
from qt_material import apply_stylesheet
from queue import Queue
import signal
import serial
import struct
import sys
import threading
from time import sleep
import websockets

# SER = serial.Serial(port="/dev/ttyACM1", baudrate=1_000_000, timeout=1)
SAMPLE_MESSAGE = cobs.encode(bytes.fromhex("AA5501")) + bytes.fromhex("00")
LOWER_MESSAGE = cobs.encode(bytes.fromhex("AA5503")) + bytes.fromhex("00")
RAISE_MESSAGE = cobs.encode(bytes.fromhex("AA5504")) + bytes.fromhex("00")
RUNNING = True
COMMAND_QUEUE = Queue()
SAMPLE_QUEUE = Queue()
MOVE_QUEUE = Queue()
# HOST = "10.0.0.155"
HOST = "192.168.0.199"
# HOST = "192.168.137.145"
TURN_SCALAR = 0.65
TURN_SCALAR_2 = 0.6


class MessageType(Enum):
    SAMPLE = 1
    LOWER = 2
    RAISE = 3
    MOVE = 4
    DRIVE = 5


class State(Enum):
    READY = 1
    MOVING = 2
    SAMPLING = 3
    LOWER = 4
    RAISE = 5


def generate_arc_trajectory(
    R=1.0,  # radius of arc (meters)
    arc_angle=np.pi,  # 180 degrees = pi radians
    v_max=0.50,  # max forward speed (m/s)
    a_max=0.50,  # max linear accel (m/s^2)
    dt=0.01,  # timestep (s)
):
    """
    Generates a smooth trapezoidal-profile trajectory for a constant-radius arc.
    Returns arrays t[], v[], w[] (times, linear speeds, angular speeds).
    """

    # Compute total arc length
    L = abs(arc_angle) * R

    # Time required to accelerate to v_max
    t_accel = v_max / a_max
    d_accel = 0.5 * a_max * t_accel**2

    # Check if we reach max velocity before needing to decelerate
    if 2 * d_accel >= L:
        # It's a triangular profile (never reaches v_max)
        v_peak = np.sqrt(a_max * L)
        t_accel = v_peak / a_max
        t_flat = 0.0
    else:
        # Full trapezoid
        d_flat = L - 2 * d_accel
        t_flat = d_flat / v_max

    t_total = 2 * t_accel + t_flat

    # Time vector
    t = np.arange(0, t_total + dt, dt)

    v = np.zeros_like(t)
    w = np.zeros_like(t)

    for i, ti in enumerate(t):
        # Acceleration phase
        if ti < t_accel:
            v[i] = a_max * ti

        # Constant velocity phase
        elif ti < (t_accel + t_flat):
            v[i] = v_max

        # Deceleration phase
        else:
            t_dec = ti - (t_accel + t_flat)
            v[i] = v_max - a_max * t_dec
            if v[i] < 0:
                v[i] = 0

        # Angular velocity from curvature: ω = v / R
        w[i] = v[i] / R
        if arc_angle < 0:  # turning right instead of left
            w[i] *= -1

    return t, v, w


class HeatmapDemo(QWidget):  # QWidget parent
    def __init__(self):
        super().__init__()

        app = QApplication.instance()
        if app is None:
            raise RuntimeError("QApplication must be constructed BEFORE HeatmapDemo")
        self.app = app

        # apply qt-material theme  ---------------------
        apply_stylesheet(self.app, theme="dark_teal.xml")

        pg.setConfigOptions(antialias=False)

        #
        # ---- Build UI container
        #
        layout = QHBoxLayout(self)
        self.setLayout(layout)

        #
        # ---- LEFT: heatmap canvas
        #
        self.win = pg.GraphicsLayoutWidget()
        self.view = self.win.addViewBox()
        self.view.setAspectLocked()
        layout.addWidget(self.win, stretch=3)

        self.img = pg.ImageItem()
        self.view.addItem(self.img)

        # starting data (100 x 100)
        self.data = np.zeros((5, 5))
        self.x_pos = 0
        self.y_pos = 0
        self.direction = 1
        self.distance = 100
        self.state = State.READY

        # colormap
        cmap = pg.colormap.get("turbo")
        self.img.setLookupTable(cmap.getLookupTable())

        #
        # ---- RIGHT: control panel
        #
        controls = QVBoxLayout()
        layout.addLayout(controls, stretch=1)

        controls.addWidget(QLabel("X"))
        self.row_input = QSpinBox()
        self.row_input.setRange(1, 1000)
        self.row_input.setValue(5)
        self.row_input.valueChanged.connect(self.resize_heatmap)
        controls.addWidget(self.row_input)

        controls.addWidget(QLabel("Y"))
        self.col_input = QSpinBox()
        self.col_input.setRange(1, 1000)
        self.col_input.setValue(5)
        self.col_input.valueChanged.connect(self.resize_heatmap)
        controls.addWidget(self.col_input)

        controls.addWidget(QLabel("Distance (cm)"))
        self.distance_input = QSpinBox()
        self.distance_input.setRange(10, 1000)
        self.distance_input.setValue(self.distance)
        self.distance_input.valueChanged.connect(self.set_distance)
        controls.addWidget(self.distance_input)

        self.sample_button = QPushButton("Sample")
        self.sample_button.clicked.connect(self.sample)
        controls.addWidget(self.sample_button)

        self.lower_button = QPushButton("Lower Sensor")
        self.lower_button.clicked.connect(self.lower_sensor)
        controls.addWidget(self.lower_button)

        self.raise_button = QPushButton("Raise Sensor")
        self.raise_button.clicked.connect(self.raise_sensor)
        controls.addWidget(self.raise_button)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_grid)
        controls.addWidget(self.run_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset)
        controls.addWidget(self.reset_button)

        controls.addStretch()

        self.plot_widget = pg.PlotWidget(title="Live Sensor Data")
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setLabel("bottom", "Sample #")
        self.plot_widget.setBackground("k")
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('c', width=2)) 

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.win, stretch=2)
        left_layout.addWidget(self.plot_widget, stretch=1)
        layout.insertLayout(0, left_layout, stretch=3)

        #
        # ---- Timer update
        #
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(33)

    # ---------------------------------------------------

    def resize_heatmap(self):
        """Called when spinboxes change"""
        rows = self.row_input.value()
        cols = self.col_input.value()
        # resize array
        self.data = np.zeros((rows, cols))
        self.x_pos = 0
        self.y_pos = 0
        self.img.setImage(self.data, autoLevels=True)

    def set_distance(self):
        self.distance = self.distance_input.value()

    def sample(self):
        COMMAND_QUEUE.put((MessageType.SAMPLE, self.x_pos, self.y_pos))

    def lower_sensor(self):
        COMMAND_QUEUE.put((MessageType.LOWER,))

    def raise_sensor(self):
        COMMAND_QUEUE.put((MessageType.RAISE,))

    def run_grid(self):
        self.state = State.SAMPLING
        COMMAND_QUEUE.put((MessageType.LOWER,))
        COMMAND_QUEUE.put((MessageType.SAMPLE, self.x_pos, self.y_pos))
        COMMAND_QUEUE.put((MessageType.RAISE,))

    def reset(self):
        self.data = np.zeros((self.row_input.value(), self.col_input.value()))
        self.x_pos = 0
        self.y_pos = 0
        self.direction = 1

    def update(self):
        if State.MOVING == self.state:
            while not MOVE_QUEUE.empty():
                move = MOVE_QUEUE.get()
                if move:
                    self.state = State.SAMPLING
                    COMMAND_QUEUE.put((MessageType.LOWER,))
                    COMMAND_QUEUE.put((MessageType.SAMPLE, self.x_pos, self.y_pos))
                    COMMAND_QUEUE.put((MessageType.RAISE,))
        else:
            while not SAMPLE_QUEUE.empty() and self.x_pos < self.row_input.value():
                turn = 0

                x, y, waveform = SAMPLE_QUEUE.get()
                self.data[x][y] = waveform.mean()
                self.plot_curve.setData(waveform)

                self.y_pos += self.direction
                if self.y_pos >= self.col_input.value():
                    turn = -1
                elif self.y_pos < 0:
                    turn = 1
                if self.y_pos >= self.col_input.value() or self.y_pos < 0:
                    self.x_pos += 1
                    self.direction *= -1
                    self.y_pos += self.direction
                if State.READY != self.state and self.x_pos < self.row_input.value():
                    if 0 != turn:
                        t, v, w = generate_arc_trajectory(
                            R=0.5,
                            arc_angle=TURN_SCALAR * np.pi,
                            v_max=0.64,
                            a_max=0.32,
                            dt=0.01,
                        )
                        for i, _ in enumerate(t):
                            COMMAND_QUEUE.put((MessageType.DRIVE, v[i], w[i] * turn))
                        sleep(0.3)
                        # Turn twice (180)?
                        t, v, w = generate_arc_trajectory(
                            R=0.5,
                            arc_angle=TURN_SCALAR_2 * np.pi,
                            v_max=0.64,
                            a_max=0.32,
                            dt=0.01,
                        )
                        for i, _ in enumerate(t):
                            COMMAND_QUEUE.put((MessageType.DRIVE, v[i], w[i] * turn))
                        COMMAND_QUEUE.put((MessageType.SAMPLE, self.x_pos, self.y_pos))
                    else:
                        self.state = State.MOVING
                        COMMAND_QUEUE.put((MessageType.MOVE, self.distance / 100, 0))
            if self.x_pos >= self.row_input.value() and State.READY != self.state:
                self.state = State.READY
        self.img.setImage(self.data, autoLevels=True)

    def run(self):
        global RUNNING
        self.show()
        self.setWindowTitle("HNSW NDE Sensor Control")
        self.resize(1200, 700)
        self.app.exec()
        RUNNING = False


def handler(signum, frame):
    global RUNNING
    RUNNING = False


# def serial_task():
#     global RUNNING
#     try:
#         while RUNNING:
#             while not COMMAND_QUEUE.empty():
#                 command = COMMAND_QUEUE.get()
#                 if 1 == command[0]:
#                     _, x, y = command
#                     SER.write(SAMPLE_MESSAGE)
#                     data = b""
#                     received = False
#                     while RUNNING and not received:
#                         if SER.in_waiting > 0:
#                             byte = SER.read(1)
#                             if 0 != len(data) or b"\x00" != byte:
#                                 data += byte
#                                 received = 0 == data[-1]
#                     data = cobs.decode(data[:-1])
#                     if 0xAA == data[0] and 0x55 == data[1] and 0x02 == data[2]:
#                         mean = sum(data) / len(data)
#                         SAMPLE_QUEUE.put((x, y, mean))
#                 elif 2 == command[0]:
#                     SER.write(LOWER_MESSAGE)
#                 elif 3 == command[0]:
#                     SER.write(RAISE_MESSAGE)

#     except Exception as e:
#         print(f"Serial error {e}")
#         RUNNING = False

#     finally:
#         SER.close()
#         print("Serial connection closed")


async def ws_task():
    global RUNNING
    state = State.READY
    try:
        async with websockets.connect(
            f"ws://{HOST}:8081/sensor"
        ) as sensor_websocket, websockets.connect(
            f"ws://{HOST}:8081/drive"
        ) as drive_websocket, websockets.connect(
            f"ws://{HOST}:8081/move"
        ) as move_websocket, websockets.connect(
            f"ws://{HOST}:8081/stepper"
        ) as stepper_websocket:
            while RUNNING:
                if State.MOVING == state:
                    response = await move_websocket.recv()
                    if len(response) > 0:
                        state = State.READY
                        MOVE_QUEUE.put(True)
                else:
                    while not COMMAND_QUEUE.empty():
                        command = COMMAND_QUEUE.get()
                        if MessageType.SAMPLE == command[0]:
                            _, x, y = command
                            await sensor_websocket.send(SAMPLE_MESSAGE)
                            data = b""
                            received = False
                            while RUNNING and not received:
                                response = await sensor_websocket.recv()
                                for byte in response:
                                    if 0 != len(data) or b"\x00" != byte:
                                        data += byte.to_bytes(1, "little")
                                        received = 0 == data[-1]
                            data = cobs.decode(data[:-1])
                            if 0xAA == data[0] and 0x55 == data[1] and 0x02 == data[2]:
                                # sleep(3)  # Artificial delay for lower/raise
                                waveform = np.frombuffer(data, dtype=np.uint8).astype(
                                    np.float32
                                )
                                SAMPLE_QUEUE.put((x, y, waveform))
                        elif MessageType.LOWER == command[0]:
                            await stepper_websocket.send(LOWER_MESSAGE)
                            await asyncio.sleep(3)
                        elif MessageType.RAISE == command[0]:
                            await stepper_websocket.send(RAISE_MESSAGE)
                            await asyncio.sleep(3)
                        elif MessageType.MOVE == command[0]:
                            state = State.MOVING
                            _, position, pose = command
                            await move_websocket.send(
                                struct.pack("<dd", position, pose)
                            )
                        elif MessageType.DRIVE == command[0]:
                            _, speed, angular_speed = command
                            await drive_websocket.send(
                                struct.pack("<dd", speed, angular_speed)
                            )
                            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"Ws error {e}")
        RUNNING = False

    finally:
        print("Ws connection closed")


def task_runner():
    asyncio.run(ws_task())
    # serial_task()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    serial_thread = threading.Thread(target=task_runner)
    serial_thread.start()

    HeatmapDemo().run()

    serial_thread.join()
