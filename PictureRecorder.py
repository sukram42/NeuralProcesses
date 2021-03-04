from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PictureRecorder:
    """
    Class to record images to see improvement of the Generator
    """
    _imgs: List
    _channel: int
    _fig: plt.Figure

    def __init__(self, figure=None, channel=1):
        self._imgs = []
        self._channel = channel
        self._fig = plt.figure() if figure is None else figure
        self._auto_fig = figure is None

    def add(self, plots):
        """Adds plottings to the video"""
        self._imgs.append(plots)

    def add_image(self, image, cmap="gray"):
        """Add an image to the recorder"""
        assert self._fig is not None
        assert not self._auto_fig

        _img = plt.imshow(image, cmap=cmap)
        self._imgs.append([_img])

    def save_movie(self, file="image.gif"):
        """
        Method to save the record
        :param file:
        :return:
        """
        print(f"Got {len(self._imgs)} frames. Will save the file now")
        ani = animation.ArtistAnimation(self._fig, self._imgs,
                                        interval=50, repeat_delay=1000)
        ani.save(file)


if __name__ == '__main__':
    recorder = PictureRecorder()
    for i in range(3):
        x = np.random.randint(0, 254, size=(28, 28))
        recorder.add_image(x)
    recorder.save_movie("testmov.mp4")
