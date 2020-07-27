import torch

from generator import Generator
import torch

import numpy as np

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QEasingCurve, Qt
import PyQt5.QtCore as QtCore
from functools import partial
from gl import glWidget

class Slider(QSlider):
    def mousePressEvent(self, e):
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), e.x(), self.width()))
    def mouseMoveEvent(self, e):
        self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), e.x(), self.width()))

class AnimateBetweenNums(QtCore.QVariantAnimation):
    def __init__(self):
        QtCore.QVariantAnimation.__init__(self)
    def updateCurrentValue(self, value):
        pass


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUI()
        self.widget = glWidget()

    def setupUI(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nz = 50
        self.noise = torch.zeros(1, self.nz, 1, 1, device=device)
        self.slider_range = 300
        self.desired_slider_value = (((np.random.rand(self.nz) - 0.5) * 2) * self.slider_range).astype(np.int32)
        self.load_network()
        self.setWindowTitle("Latent voyage")

        self.random_button = QPushButton("Randomize")
        self.random_button.clicked.connect(self.randomize_latent_vector)
        
        self.widget_gl = glWidget()

        box_rotation = QVBoxLayout()
        box_rotation.setSpacing(20)
        for i in ['x','y','z']:
            slider = Slider(Qt.Horizontal)
            slider.setRange(-180,180)
            slider.valueChanged.connect(partial(self.widget_gl.set_model, axis=i))
            box_rotation.addWidget(slider)
        widget_rotation = QWidget()
        widget_rotation.setLayout(box_rotation)

        box_latent = QVBoxLayout()
        box_latent.setSpacing(20)
        self.sliders = []
        for i in range(0, self.nz):
            slider = Slider(Qt.Horizontal)
            self.sliders.append(slider)
            slider.setRange(-self.slider_range, self.slider_range)
            slider.valueChanged.connect(partial(self.set_noise_value, idx=i))
            box_latent.addWidget(slider)
        widget_latent = QWidget()
        widget_latent.setLayout(box_latent)
        
        # Animation for randomize button
        self.animation_speed = 1000
        self.animation_steps = 60
        self.anim = AnimateBetweenNums()
        self.anim.setDuration(self.animation_speed)
        self.anim.valueChanged.connect(self.set_animtaion)
        self.anim.setEasingCurve(QEasingCurve.InOutCirc)

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedWidth(512)
        self.scroll.setWidget(widget_latent)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("image"))
        layout.addWidget(self.widget_gl)
        layout.addWidget(self.random_button)
        layout.addWidget(QLabel("Rotation(x,y,z)"))
        layout.addWidget(widget_rotation)
        layout.addWidget(QLabel("Latent"))
        layout.addWidget(self.scroll)
        self.setGeometry(100,100,512,1000)
        self.setLayout(layout)
        self.show()
        
    def set_noise_value(self, value, idx):
        self.noise[:,idx,:,:] = value/100.0
        self.set_image()

    def set_image(self):
        rgb_image = np.clip((self.get_image() + 1.0) * 0.5, 0, 1) * 255
        rgb_image = rgb_image.astype('uint8')
        self.widget_gl.set_texture(rgb_image)

    def lerp(self, a,b,t):
        return a + t * (b - a)    

    def set_animtaion(self, value):
        current_value = self.slider_value_list[value]
        for i, slider in enumerate(self.sliders):
            slider.setValue(int(current_value[i]))
        
    def randomize_latent_vector(self):
        current_slider_value = np.array([slider.value() for slider in self.sliders]).astype(np.int32)
        self.desired_slider_value = (((np.random.rand(self.nz) - 0.5) * 2) * self.slider_range).astype(np.int32)
        self.slider_value_list = [self.lerp(current_slider_value, self.desired_slider_value, i/self.animation_steps) for i in range(self.animation_steps)]
        self.anim.stop()
        self.anim.setStartValue(0)
        self.anim.setEndValue(self.animation_steps-1)
        self.anim.start()

    def load_network(self):
        # Number of GPUs available. Use 0 for CPU mode.
        ngpu = 1

        saved_model_dir = "trained_network/Generator.pth"

        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

        # Create the generator
        self.netG = Generator(ngpu).to(device)
        
        self.netG.load_state_dict(torch.load(saved_model_dir, map_location=device))
        self.netG.eval()

    def get_image(self):
        with torch.no_grad():
            pic = self.netG(self.noise).detach().cpu().numpy()
            pic = pic[0, :, :, :]
            pic = pic.transpose(1,2,0)
            return pic
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    loop=QtCore.QTimer()
    loop.timeout.connect(window.widget_gl.update)
    loop.start(0)

    sys.exit(app.exec_())
