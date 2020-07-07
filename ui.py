import numpy as np
import os
import pickle
import json
from types import SimpleNamespace
import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.scatter import Scatter
from PIL import Image as im

from power_planner.utils.utils import (
    get_distance_surface, time_test_csv, compute_pylon_dists
)
from power_planner.graphs.implicit_lg import ImplicitLG
from power_planner.alternative_paths import AlternativePaths
from power_planner.ksp import KSP


class ImageFromArray(Widget):

    def __init__(self, width, height, **kwargs):
        super(ImageFromArray, self).__init__(**kwargs)
        self.max_width = width
        self.max_height = height
        # img = np.random.rand(width, height, 3) * 255
        img = np.zeros((width, height, 3))
        self.set_array(img)

    def set_array(self, img_in):
        img_in = np.flip(np.swapaxes(img_in, 1, 0), axis=0).astype(np.uint8)
        h, w, _ = img_in.shape
        # compute how much we have to resize it to make it fit bounds
        ratio_resize = max([w / self.max_width, h / self.max_height])
        # convert to PIL Image - Note: axes are swapped!
        img_in = im.fromarray(img_in)
        new_img_size = (int(w / ratio_resize), int(h / ratio_resize))
        # resize
        img = np.array(img_in.resize(new_img_size, resample=im.BILINEAR))
        # make texture
        texture = Texture.create(size=new_img_size)
        texture.blit_buffer(img.flatten(), colorfmt='rgb', bufferfmt='ubyte')

        w_img = Image(size=new_img_size, texture=texture)
        self.add_widget(w_img)


class ResizableDraggablePicture(Scatter):

    def on_touch_down(self, touch):
        # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.scale < 10:
                    self.scale = self.scale * 1.1
            elif touch.button == 'scrollup':
                if self.scale > 1:
                    self.scale = self.scale * 0.8
        # If some other kind of "touch": Fall back on Scatter's behavior
        else:
            super(ResizableDraggablePicture, self).on_touch_down(touch)


class DemoApp(App):

    def build(self):
        # Defaults
        self.SCALE_PARAM = 5

        right_bar_size = .3
        slider_bar_size = .2
        superBox = BoxLayout(orientation='vertical')

        # Data read field
        data_box = BoxLayout(
            orientation='horizontal', size_hint=(1, None), height=30
        )
        self.filepath = TextInput(
            hint_text="data/ch_data_1_5.dat",
            size_hint=(1 - right_bar_size, 1)
        )
        self.filepath.text = "data/ch_data_1_5.dat"  # set default
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
            hint_text="data/ch_config.json", size_hint=(1 - right_bar_size, 1)
        )
        self.json_fp.text = "data/ch_config.json"  # set manually here
        self.load_json_but = Button(
            text='Load configuration',
            on_press=self.load_json,
            size_hint=(right_bar_size, 1)
        )
        self.load_json_but.disabled = True
        config_box.add_widget(self.json_fp)
        config_box.add_widget(self.load_json_but)

        # Declare initial status
        self.data_loaded = False
        self.json_loaded = False

        # Sliders
        self.slider_box = BoxLayout(
            orientation='vertical',
            spacing=20,
            size_hint=(None, 1),
            width=Window.width * slider_bar_size
        )

        # additional buttons
        button_box = BoxLayout(
            orientation='vertical',
            spacing=50,
            size_hint=(None, 1),
            width=Window.width * right_bar_size
        )
        # # Sliders for weights
        # angle_label = Label(text="Angle weighting")
        # self.angle_slider = Slider(min=0, max=1)
        # self.angle_slider.value = 0.5
        # edge_label = Label(text="Edge weighting")
        # self.edge_slider = Slider(min=0, max=1)
        # self.edge_slider.value = 0.5
        # button_box.add_widget(edge_label)
        # button_box.add_widget(self.edge_slider)
        # button_box.add_widget(angle_label)
        # button_box.add_widget(self.angle_slider)

        # Define right side buttons
        self.single_button = Button(
            text="Single shortest path",
            on_press=self.single_sp,
            size=(Window.width * right_bar_size, 30)
        )
        self.sp_tree_button = Button(
            text="Shortest path trees",
            on_press=self.sp_tree,
            size=(Window.width * right_bar_size, 30)
        )
        # self.sp_button = Button(
        #     text="Shortest path",
        #     on_press=self.shortest_path,
        #     size=(Window.width * right_bar_size, 30)
        # )
        self.ksp_button = Button(
            text="KSP",
            on_press=self.ksp,
            size=(Window.width * right_bar_size, 30)
        )
        # Add to widget
        for button in [
            self.single_button, self.sp_tree_button, self.ksp_button
        ]:
            button.disabled = True
            button_box.add_widget(button)

        # make horizontal box with canvas and buttons
        canv_box = BoxLayout(orientation='horizontal')
        self.img_widget = ImageFromArray(400, 500)
        # for scroll function, add the comment - but not working well
        self.scatter_widget = Scatter()  # ResizableDraggablePicture()
        self.scatter_widget.add_widget(self.img_widget)
        canv_box.add_widget(self.scatter_widget)
        canv_box.add_widget(self.slider_box)
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
                    self.instance, self.edge_inst, self.instance_corr,
                    self.config
                ) = data
            print(self.instance.shape)
            self.disp_inst = np.moveaxis(self.instance, 0, -1)[:, :, :3] * 255
            print(self.disp_inst.shape)
            self.img_widget.set_array(self.disp_inst)
            self.graph = ImplicitLG(self.instance, self.instance_corr)
            self.cfg = self.config.graph
            self.SCALE_PARAM = int(self.filepath.text.split(".")[0][-1])
            # enable button
            self.load_json_but.disabled = False
            self.sp_tree_button.disabled = False
            self.single_button.disabled = False

    def init_slider_box(self, instance):
        # Sliders for angle and edges
        angle_label = Label(text="Angle weight")
        self.angle_slider = Slider(min=0, max=1)
        edge_label = Label(text="Edge weight")
        self.edge_slider = Slider(min=0, max=1)
        self.slider_box.add_widget(edge_label)
        self.slider_box.add_widget(self.edge_slider)
        self.slider_box.add_widget(angle_label)
        self.slider_box.add_widget(self.angle_slider)
        self.angle_slider.value = self.cfg.ANGLE_WEIGHT
        self.edge_slider.value = self.cfg.EDGE_WEIGHT
        # make one slider for each one
        normed_weights = np.asarray(self.cfg.class_weights
                                    ) / np.sum(self.cfg.class_weights)
        self.weight_sliders = []
        for (name, weight) in zip(self.cfg.layer_classes, normed_weights):
            label = Label(text=name)
            slider = Slider(min=0, max=1)
            slider.value = float(weight)
            self.weight_sliders.append(slider)
            self.slider_box.add_widget(label)
            self.slider_box.add_widget(slider)

    def load_json(self, instance):
        # with open(self.json_fp.text, "r") as infile:
        #     self.cfg_dict = json.load(infile)
        #     self.cfg = SimpleNamespace(**self.cfg_dict)
        #     (self.cfg.PYLON_DIST_MIN,
        #      self.cfg.PYLON_DIST_MAX) = compute_pylon_dists(
        #          self.cfg.PYLON_DIST_MIN, self.cfg.PYLON_DIST_MAX,
        #          self.cfg.RASTER, self.SCALE_PARAM
        #      )
        self.init_slider_box(instance)

    def single_sp(self, instance, buffer=1):
        new_class_weights = [slider.value for slider in self.weight_sliders]
        self.cfg.class_weights = new_class_weights
        # new_img = (np.random.rand(1000, 400, 3) * 150)
        # self.img_widget.set_array(new_img)
        path, _, _ = self.graph.single_sp(self.edge_inst, **vars(self.cfg))
        plotted_inst = self.path_plotter(
            self.disp_inst.copy(), path, 255, buffer=buffer
        )
        self.img_widget.set_array(plotted_inst)
        print("Done single shortest path")

    def sp_tree(self, instance, buffer=1):
        new_class_weights = [slider.value for slider in self.weight_sliders]
        self.cfg.class_weights = new_class_weights
        self.cfg.ANGLE_WEIGHT = self.angle_slider.value
        self.cfg.EDGE_WEIGHT = self.edge_slider.value
        # set edge cost (must be repeated because of angle weight)
        path, _, _ = self.graph.sp_trees(self.edge_inst, **vars(self.cfg))
        # plot the path
        plotted_inst = self.path_plotter(
            self.disp_inst.copy(), path, 255, buffer=buffer
        )
        self.img_widget.set_array(plotted_inst)
        # enable KSP
        self.ksp_button.disabled = False
        print("Done shortest path trees")

    def ksp(self, instance, buffer=2):
        ksp = KSP(self.graph)
        ksp_output = ksp.max_vertex_ksp(5, min_dist=15)
        paths = [k[0] for k in ksp_output]
        plotted_inst = self.disp_inst.copy()
        for i in range(len(paths) - 1, -1, -1):
            path = paths[i]
            val = 255 - i * 50
            plotted_inst = self.path_plotter(
                plotted_inst, path, val, buffer=buffer
            )
        self.img_widget.set_array(plotted_inst)
        print("ksp done")

    def path_plotter(self, plotted_inst, path, val, buffer=1):
        for (x, y) in path:
            plotted_inst[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                         1] = [val, val, val]
        return plotted_inst


#

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
