from plots.histograms import update_histograms
from plots.projection_plots import update_projection_graphs


def update_plots(self):
    """
    Updates the projection graphs and histograms for the current image.
    """
    self.gray_image = self.convert_to_grayscale()
    update_projection_graphs(self)
    update_histograms(self)