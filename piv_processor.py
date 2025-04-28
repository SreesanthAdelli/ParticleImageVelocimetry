import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from skimage.io import imread

class PIVProcessor:
    def __init__(self, img1_path, img2_path, win_size=32):
        self.img1 = imread(img1_path)
        self.img2 = imread(img2_path)
        self.win_size = win_size

    def compute_velocity_field(self):
        a, b = self.img1, self.img2
        win_size = self.win_size

        ys = np.arange(0, a.shape[0] - win_size, win_size)
        xs = np.arange(0, a.shape[1] - win_size, win_size)
        dys = np.zeros((len(ys), len(xs)))
        dxs = np.zeros((len(ys), len(xs)))

        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                int_win = a[y : y + win_size, x : x + win_size]
                search_win = b[y : y + win_size, x : x + win_size]
                cross_corr = correlate(
                    search_win - search_win.mean(), int_win - int_win.mean(), method="fft"
                )
                peak_y, peak_x = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
                dy, dx = (peak_y - win_size + 1), (peak_x - win_size + 1)
                dys[iy, ix], dxs[iy, ix] = dy, dx

        # Position vectors at window centers
        self.xs = xs + win_size / 2
        self.ys = ys + win_size / 2
        self.dxs = dxs
        self.dys = dys

    def plot_velocity_field(self):
        norm_drs = np.sqrt(self.dxs**2 + self.dys**2)

        fig, ax = plt.subplots(figsize=(6, 6))
        q = ax.quiver(
            self.xs,
            self.ys[::-1],
            self.dxs,
            -self.dys,
            norm_drs,
            cmap="plasma",
            angles="xy",
            scale_units="xy",
            scale=0.25,
        )
        ax.set_aspect("equal")
        plt.title("PIV Result")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(q, ax=ax, label="Displacement magnitude")
        plt.show()
