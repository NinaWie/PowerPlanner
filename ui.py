import numpy as np
import os
import pickle
import json
from types import SimpleNamespace
import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.lang import Builder


class Test(Widget):

    def __init__(self, **kwargs):
        super(Test, self).__init__(**kwargs)

        img = (np.random.rand(400, 400, 3) * 255)
        self.set_array(img)

    def set_array(self, img):
        img = img.astype(np.uint8)
        w, h, _ = img.shape
        texture = Texture.create(size=(w, h))
        texture.blit_buffer(img.flatten(), colorfmt='rgb', bufferfmt='ubyte')

        w_img = Image(size=(w, h), texture=texture)
        self.add_widget(w_img)


class DemoApp(App):

    def build(self):
        right_bar_size = .3
        superBox = BoxLayout(orientation='vertical')

        # Data read field
        data_box = BoxLayout(
            orientation='horizontal', size_hint=(1, None), height=30
        )
        self.filepath = TextInput(
            hint_text="data/data_dump_5.dat",
            size_hint=(1 - right_bar_size, 1)
        )
        self.filepath.text = "data/data_dump_5.dat"  # set default
        self.load_data = Button(
            text='Load data',
            on_press=self.loadData,
            size_hint=(right_bar_size, 1)
        )
        data_box.add_widget(self.filepath)
        data_box.add_widget(self.load_data)

        # configuration json
        config_box = BoxLayout(
            orientation='horizontal', size_hint=(1, None), height=30
        )
        self.json_fp = TextInput(
            hint_text="config.json", size_hint=(1 - right_bar_size, 1)
        )
        self.json_fp.text = "config.json"  # set manually here
        self.load_json_but = Button(
            text='Load configuration',
            on_press=self.load_json,
            size_hint=(right_bar_size, 1)
        )
        config_box.add_widget(self.json_fp)
        config_box.add_widget(self.load_json_but)

        # additional buttons
        button_box = BoxLayout(
            orientation='vertical',
            spacing=80,
            size_hint=(None, 1),
            width=Window.width * right_bar_size
        )
        for button_text in ["initialize", "build_graph", "shortest_path"]:
            but_tmp = Button(
                text=button_text,
                on_press=eval("self." + button_text),
                size=(Window.width * right_bar_size, 30)
            )
            button_box.add_widget(but_tmp)

        # make horizontal box with canvas and buttons
        canv_box = BoxLayout(orientation='horizontal')
        self.img_widget = Test()
        canv_box.add_widget(self.img_widget)
        canv_box.add_widget(button_box)

        # add to final box
        superBox.add_widget(data_box)
        superBox.add_widget(config_box)
        superBox.add_widget(canv_box)

        return superBox

    def loadData(self, instance):
        # self.filepath.text = str(os.path.exists(self.filepath.text))
        if os.path.exists(self.filepath.text):
            with open(self.filepath.text, "rb") as infile:
                data = pickle.load(infile)
                (
                    self.instance, self.instance_corr, self.start_inds,
                    self.dest_inds
                ) = data.data
            print(self.instance.shape)

    def load_json(self, instance):
        with open(self.json_fp.text, "r") as infile:
            cfg_dict = json.load(infile)  # Config(SCALE_PARAM)
            cfg = SimpleNamespace(**cfg_dict)
            cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX = compute_pylon_dists(
                cfg.PYLON_DIST_MIN, cfg.PYLON_DIST_MAX, cfg.RASTER, 5
            )

    def initialize(self, instance):
        new_img = (np.random.rand(1000, 400, 3) * 150)
        self.img_widget.set_array(new_img)
        print("initialize button")

    def build_graph(self, instance):
        print("build_graph button")

    def shortest_path(self, instance):
        print("shortest_path button")


# Builder.load_string(
#     """

# <KivyButton>:

#     Button:

#         text: "Hello Button!"

#         size_hint: .12, .12

#         Image:

#             source: 'images.jpg'

#             center_x: self.parent.center_x

#             center_y: self.parent.center_y

# """
# )

# # code to disable button
# def disable(self, instance, *args):

#     instance.disabled = True

# def update(self, instance, *args):

#     instance.text = "I am Disabled!"

# def build(self):

#     mybtn = Button(
#         text="Click me to disable", pos=(300, 350), size_hint=(.25, .18)
#     )

#     mybtn.bind(on_press=partial(self.disable, mybtn))

#     mybtn.bind(on_press=partial(self.update, mybtn))

#     return mybtn

## RUN MATPLOTLIB:
# plt.plot([10, 30, 20, 45])

# class MyApp(App):

#     def build(self):
#     box = BoxLayout()
#     box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
#     return box
# https://github.com/kivy-garden/garden.matplotlib/blob/master/examples/test_plt.py
# import matplotlib
# matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')

if __name__ == '__main__':
    DemoApp().run()
