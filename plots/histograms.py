def update_histograms(self):
    """
    Updates histograms for the grayscale image and RGB channels.
    """
    if self.gray_image is None or self.numpy_image is None:
        return

    self.hist_fig.clf()

    ax_gray = self.hist_fig.add_subplot(2, 2, 1)
    ax_red = self.hist_fig.add_subplot(2, 2, 2)
    ax_green = self.hist_fig.add_subplot(2, 2, 3)
    ax_blue = self.hist_fig.add_subplot(2, 2, 4)
    self.hist_fig.set_facecolor("#C1C1C1")

    font_settings = {'fontsize': 8, 'fontfamily': 'Consolas'}

    ax_gray.hist(self.gray_image.ravel(), bins=256, range=(0, 255), color='black')
    ax_gray.set_title("Gray Scale Intensity", fontdict=font_settings)
    ax_gray.set_facecolor("#C1C1C1")
    ax_gray.yaxis.set_visible(False)

    ax_red.hist(self.tmp_numpy_image[:, :, 0].ravel(), bins=256, range=(0, 255), color='red')
    ax_red.set_title("Red Channel Intensity", fontdict=font_settings)
    ax_red.set_facecolor("#C1C1C1")
    ax_red.yaxis.set_visible(False)

    ax_green.hist(self.tmp_numpy_image[:, :, 1].ravel(), bins=256, range=(0, 255), color='green')
    ax_green.set_title("Green Channel Intensity", fontdict=font_settings)
    ax_green.set_facecolor("#C1C1C1")
    ax_green.yaxis.set_visible(False)

    ax_blue.hist(self.tmp_numpy_image[:, :, 2].ravel(), bins=256, range=(0, 255), color='blue')
    ax_blue.set_title("Blue Channel Intensity", fontdict=font_settings)
    ax_blue.set_facecolor("#C1C1C1")
    ax_blue.yaxis.set_visible(False)

    for ax in [ax_gray, ax_red, ax_green, ax_blue]:
        ax.tick_params(axis='both', labelsize=6)

    self.hist_fig.tight_layout()
    self.hist_canvas.draw()
