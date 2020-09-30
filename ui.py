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
from kivy.uix.popup import Popup
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
        # img_in = img_in[20:-20, 20:-20, :3]
        self.current_in_img = img_in
        img_in = np.flip(np.swapaxes(img_in, 1, 0), axis=0).astype(np.uint8)
        h, w, _ = img_in.shape
        # compute how much we have to resize it to make it fit bounds
        ratio_resize = max([w / self.max_width, h / self.max_height])
        # convert to PIL Image - Note: axes are swapped!
        img_in = im.fromarray(img_in)
        new_img_size = (int(w / ratio_resize), int(h / ratio_resize))
        # resize
        img = np.array(img_in.resize(new_img_size, resample=im.BILINEAR))
        self.current_img = img
        # make texture
        texture = Texture.create(size=new_img_size)
        texture.blit_buffer(img.flatten(), colorfmt='rgb', bufferfmt='ubyte')

        self.take_size = new_img_size
        w_img = PressableImage(size=new_img_size, texture=texture)
        self.add_widget(w_img)


class PressableImage(Image):

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print("pressed", touch.pos)
            # touch_str = str(round(touch.pos[0])
            #  ) + "_" + str(round(touch.pos[1]))
            touch_str = str(
                np.around(
                    self.parent.current_img[int(touch.pos[0]),
                                            int(touch.pos[1])], 1
                )
            )
            popupWindow = Popup(
                title="Resistance",
                content=Label(text=touch_str),
                size_hint=(None, None),
                size=(150, 100)
            )
            popupWindow.open()
        # return super(PressableImage, self).on_touch_down(touch)


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
        self.sliders_initialized = 0

        right_bar_size = .3
        slider_bar_size = .2
        superBox = BoxLayout(orientation='vertical')

        # Data read field
        data_box = BoxLayout(
            orientation='horizontal', size_hint=(1, None), height=30
        )
        self.filepath = TextInput(
            hint_text="data/test_data_1_2.dat",
            size_hint=(1 - right_bar_size, 1)
        )
        self.filepath.text = "data/test_data_1_2.dat"  # set default
        self.load_data = Button(
            text='Load data',
            on_press=self.loadData,
            size_hint=(right_bar_size + 0.05, 1)
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
        # config_box.add_widget(self.json_fp)
        # config_box.add_widget(self.load_json_but)

        # Declare initial status
        self.data_loaded = False
        self.json_loaded = False

        # Sliders
        self.slider_box = BoxLayout(
            orientation='vertical', width=Window.width * slider_bar_size
        )

        # additional buttons
        button_box = BoxLayout(
            orientation='vertical', width=Window.width * right_bar_size
        )

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
        self.alternative_button = Button(
            text="Informed path routing",
            on_press=self.rect_popup,
            size=(Window.width * right_bar_size, 30)
        )
        # Add to widget
        for button in [
            self.single_button, self.sp_tree_button, self.ksp_button,
            self.alternative_button
        ]:
            button.disabled = True
            button_box.add_widget(button)

        # make horizontal box with canvas and buttons
        canv_box = BoxLayout(orientation='horizontal')
        self.img_widget = ImageFromArray(600, 500)
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

    def _mark_start_dest(self, buffer=2):
        (x, y) = tuple(self.config.graph.start_inds)
        self.disp_inst[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                       1] = [255, 255, 0]
        (x, y) = tuple(self.config.graph.dest_inds)
        self.disp_inst[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                       1] = [255, 255, 0]

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
            # disp instance is with RGB and overlayed with corridor
            self.disp_inst = (
                np.moveaxis(self.instance, 0, -1)[:, :, :3] * 255
            ) * np.expand_dims(self.instance_corr, 2)
            self._mark_start_dest()
            print(self.disp_inst.shape)
            self.img_widget.set_array(self.disp_inst)
            self.graph = ImplicitLG(
                self.instance,
                self.instance_corr,
                edge_instance=self.edge_inst,
            )
            self.cfg = self.config.graph
            self.SCALE_PARAM = int(self.filepath.text.split(".")[0][-1])
            # enable button
            self.load_json_but.disabled = False
            self.sp_tree_button.disabled = False
            self.single_button.disabled = False
            # init sliders
            self.init_slider_box(instance)

    def init_slider_box(self, instance):
        if not self.sliders_initialized:
            # Sliders for angle and edges
            angle_label = Label(text="Angle weight")
            self.angle_slider = Slider(min=0, max=1)
            edge_label = Label(text="Edge weight")
            self.edge_slider = Slider(min=0, max=1)
            self.slider_box.add_widget(edge_label)
            self.slider_box.add_widget(self.edge_slider)
            self.slider_box.add_widget(angle_label)
            self.slider_box.add_widget(self.angle_slider)
            # make one slider for each category
            self.weight_sliders = []
            for name in self.cfg.layer_classes:
                label = Label(text=name)
                slider = Slider(min=0, max=1)
                self.weight_sliders.append(slider)
                self.slider_box.add_widget(label)
                self.slider_box.add_widget(slider)

            # set it to initialized
            self.sliders_initialized = 1

        # UPDATE VALUES ACCORDING TO CONFIG
        self.angle_slider.value = self.cfg.angle_weight
        self.edge_slider.value = self.cfg.edge_weight
        normed_weights = np.asarray(self.cfg.class_weights
                                    ) / np.sum(self.cfg.class_weights)
        for i in range(len(normed_weights)):
            self.weight_sliders[i].value = float(normed_weights[i])

    def load_json(self, instance):
        # with open(self.json_fp.text, "r") as infile:
        #     self.cfg_dict = json.load(infile)
        #     self.cfg = SimpleNamespace(**self.cfg_dict)
        #     (self.cfg.PYLON_DIST_MIN,
        #      self.cfg.PYLON_DIST_MAX) = compute_pylon_dists(
        #          self.cfg.PYLON_DIST_MIN, self.cfg.PYLON_DIST_MAX,
        #          self.cfg.RASTER, self.SCALE_PARAM
        #      )
        # self.init_slider_box(instance)
        pass

    def single_sp(self, instance, buffer=1):
        new_class_weights = [slider.value for slider in self.weight_sliders]
        self.cfg.class_weights = new_class_weights
        self.cfg.angle_weight = self.angle_slider.value
        self.cfg.edge_weight = self.edge_slider.value
        # new_img = (np.random.rand(1000, 400, 3) * 150)
        # self.img_widget.set_array(new_img)
        path, _, _ = self.graph.single_sp(**vars(self.cfg))
        plotted_inst = self.path_plotter(
            self.disp_inst.copy(), path, [255, 255, 255], buffer=buffer
        )
        self.img_widget.set_array(plotted_inst)
        print("Done single shortest path")

    def sp_tree(self, instance, buffer=1):
        new_class_weights = [slider.value for slider in self.weight_sliders]
        self.cfg.class_weights = new_class_weights
        self.cfg.angle_weight = self.angle_slider.value
        self.cfg.edge_weight = self.edge_slider.value
        # set edge cost (must be repeated because of angle weight)
        path, _, _ = self.graph.sp_trees(**vars(self.cfg))
        # plot the path
        plotted_inst = self.path_plotter(
            self.disp_inst.copy(), path, [255, 255, 255], buffer=buffer
        )
        self.img_widget.set_array(plotted_inst)
        # enable KSP
        self.ksp_button.disabled = False
        self.alternative_button.disabled = False
        print("Done shortest path trees")

    def ksp(self, instance, buffer=1):
        ksp = KSP(self.graph)
        ksp_output = ksp.laplace(5, thresh=20)
        paths = [k[0] for k in ksp_output]
        plotted_inst = self.disp_inst.copy()
        for i in range(len(paths) - 1, -1, -1):
            path = paths[i]
            val = 255 - i * 30
            plotted_inst = self.path_plotter(
                plotted_inst, path, [val, val, val], buffer=buffer
            )
        self.img_widget.set_array(plotted_inst)
        print("ksp done")

    def paint_rectangle(self, instance, buffer=2):
        try:
            ul_x = int(self.rect_text[0].text)
            ul_y = int(self.rect_text[1].text)
            br_x = int(self.rect_text[2].text)
            br_y = int(self.rect_text[3].text)
        except ValueError:
            print("error: not all values given")
            return
        copied_inst = self.disp_inst.copy()
        copied_inst[ul_x:br_x, ul_y - buffer:ul_y + buffer + 1] = [255, 0, 0]
        copied_inst[ul_x:br_x, br_y - buffer:br_y + buffer + 1] = [255, 0, 0]
        copied_inst[ul_x - buffer:ul_x + buffer + 1, ul_y:br_y] = [255, 0, 0]
        copied_inst[br_x - buffer:br_x + buffer + 1, ul_y:br_y] = [255, 0, 0]
        self.rect = (ul_x, br_x, ul_y, br_y)
        self.img_widget.set_array(copied_inst)
        self.but_dismiss.disabled = False

    def rect_popup(self, instance):

        box = BoxLayout(orientation='vertical', height=200)
        text_labels = [
            "Upper left X", "Upper left Y", "Lower right X", "lower right Y"
        ]
        text = text_labels  # [150, 250, 200, 300]
        self.rect_text = []
        for (l, t) in zip(text_labels, text):
            t_in = TextInput(hint_text=l)
            t_in.text = str(t)
            self.rect_text.append(t_in)
            box.add_widget(t_in)

        # add buttons
        but = Button(text='Paint rectangle', on_press=self.paint_rectangle)
        self.but_dismiss = Button(text='Finish', on_press=self.alternative)
        self.but_dismiss.disabled = True
        box.add_widget(but)
        box.add_widget(self.but_dismiss)
        # define popup
        self.popupWindow = Popup(
            title="Set rectangle",
            content=box,
            size_hint=(None, None),
            size=(300, 300)
        )
        self.popupWindow.open()

    def alternative(self, instance, buffer=2):
        self.popupWindow.dismiss()
        print("rect", self.rect)
        alt = AlternativePaths(self.graph)
        replacement_path, _, _ = alt.path_through_window(*self.rect)
        plot_inst = self.img_widget.current_in_img.copy()
        plotted_inst = self.path_plotter(
            plot_inst, replacement_path, [255, 0, 0], buffer=buffer
        )
        self.img_widget.set_array(plotted_inst)
        print("replacement done")

    def path_plotter(self, plotted_inst, path, col, buffer=1):
        for (x, y) in path:
            plotted_inst[x - buffer:x + buffer + 1, y - buffer:y + buffer +
                         1] = col
        return plotted_inst


if __name__ == '__main__':
    DemoApp().run()
