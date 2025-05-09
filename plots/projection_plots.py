import numpy as np


def update_projection_graphs(self):
    """
    Updates both vertical and horizontal projection graphs.
    """
    update_v_projection_graph(self)
    update_h_projection_graph(self)



def update_v_projection_graph(self):
    """
    Updates the vertical projection graph of the image.
    """
    if self.mod_image:
        img_data = self.gray_image

        vertical_projection = np.sum(img_data, axis=0)

        if self.mod_image_label.pixmap():
            img_width = self.mod_image_label.pixmap().width()
            self.left_lower_left_v_projection_panel.setFixedWidth(img_width)

        self.v_projection_canvas.figure.clf()

        ax = self.v_projection_canvas.figure.add_subplot(111)
        ax.plot(vertical_projection, linewidth=2, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.v_projection_canvas.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_facecolor("#C1C1C1")
        self.v_projection_canvas.figure.set_facecolor("#C1C1C1")

        self.v_projection_canvas.draw()
        self.v_projection_layout.addWidget(self.v_projection_canvas)


def update_h_projection_graph(self):
    """
    Updates the horizontal projection graph of the image.
    """
    if self.mod_image:
        img_data = self.gray_image

        horizontal_projection = np.sum(img_data, axis=1)

        if self.mod_image_label.pixmap():
            img_height = self.mod_image_label.pixmap().height()
            self.left_lower_h_projection_panel.setFixedHeight(img_height)

        self.h_projection_canvas.figure.clf()
        ax = self.h_projection_canvas.figure.add_subplot(111)
        ax.plot(horizontal_projection, range(len(horizontal_projection)), linewidth=2, color='red')

        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self.h_projection_canvas.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_facecolor("#C1C1C1")
        self.h_projection_canvas.figure.set_facecolor("#C1C1C1")

        self.h_projection_canvas.draw()
        self.h_projection_layout.addWidget(self.h_projection_canvas)
