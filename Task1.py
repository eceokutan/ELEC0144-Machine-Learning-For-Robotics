# CODE MAP of Task1.py
#   Settings           darkFactor, mirror, median window size, RGB thresholds of each fruit
#   Load image         read Fruits.jpg, darken (multiply, round to 8 bits), mirror
#   Normalise          pixel point processing: scale so the brightest value is 255 again
#   Thresholding       pixel point processing: keep pixels whose R, G, B lie in each fruit's range
#   Median filter      pixel group processing: median of each k x k window removes speckle noise
#   Largest blob       blob = connected component; keep only the biggest one of each mask
#   Box & centroids    bounding box; centroid a) midpoint of min/max, b) moments M10/M00, M01/M00
#   Stage images       black-and-white masks of every stage, and darkened vs normalised
#   Plot               boxes, both centroids and names drawn on the image
#   AI use             Claude Opus 5.5 (claude-opus-5-5), default setting

# %% Settings
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

darkFactor = 0.8      # every pixel is multiplied by this, then rounded to 8 bits
mirror = False         # True: flip the image left-to-right

MEDIAN_SIZE = 7       # the median filter uses a MEDIAN_SIZE x MEDIAN_SIZE window (odd)
SHOW_STAGES = True    # also plot the black-and-white masks of every stage

# Thresholds on the three colour layers (after normalising), as (min, max).
# A pixel belongs to the fruit only if its R, G and B are all inside the ranges.
# The values were read off the image: e.g. the capsicum is very red (R > 235)
# with almost no blue (B < 8), whereas the orange wood always has some blue.
THRESHOLDS = {
    "Apple":    {"R": (190, 255), "G": (0, 100),   "B": (40, 160)},
    "Capsicum": {"R": (235, 255), "G": (60, 150),  "B": (0, 8)},
    "Lemon":    {"R": (225, 255), "G": (150, 255), "B": (0, 110)},
}
COLOURS = {"Apple": "cyan", "Capsicum": "lime", "Lemon": "magenta"}

# %% Load image
here = os.path.dirname(os.path.abspath(__file__))
I = plt.imread(os.path.join(here, "Fruits.jpg"))[..., :3]
I = I.astype(float)                       # double, so sums do not saturate at 255

I = np.round(I * darkFactor)              # darken: multiply, round to 8 bits
if mirror:
    I = I[:, ::-1, :]                     # flip left-to-right
I_show = I.astype(np.uint8)               # copy for displaying at the end
m, n = I.shape[:2]                        # m rows (y), n columns (x)

# %% Normalise
# Darkening multiplied every pixel by darkFactor, so the brightest value in
# the image dropped from 255 to about 255*darkFactor. Multiplying every pixel
# by 255 / (brightest value) undoes this, so the same thresholds work for any
# dark factor. It is a pixel point process: each pixel only uses its own value.
I = I * 255 / I.max()
IRed, IGreen, IBlue = I[:, :, 0], I[:, :, 1], I[:, :, 2]


# %% Thresholding
def threshold(t):
    """Binary image: 1 where all three colour layers are inside the ranges."""
    return ((IRed >= t["R"][0]) & (IRed <= t["R"][1]) &
            (IGreen >= t["G"][0]) & (IGreen <= t["G"][1]) &
            (IBlue >= t["B"][0]) & (IBlue <= t["B"][1]))


# %% Median filter
def median_filter(BW, k):
    """Median filter with a k x k window: each output
    pixel is the median of the k*k pixels around it. The k*k shifted copies of
    the image are stacked so the median of every window is taken at once."""
    p = k // 2
    padded = np.pad(BW.astype(float), p)              # zeros around the border
    windows = np.stack([padded[i:i + m, j:j + n]
                        for i in range(k) for j in range(k)])
    return np.median(windows, axis=0) > 0.5


# %% Largest blob
def largest_blob(BW):
    """A blob is a set of connected (adjacent) white pixels. Starting from each
    white pixel not yet visited, collect every white pixel joined to it
    (8 neighbours), and keep the blob with the most pixels. This removes any
    small stray blobs that would otherwise move the min/max of the box."""
    visited = np.zeros_like(BW, dtype=bool)
    best, best_size = [], 0
    for y0, x0 in zip(*np.nonzero(BW)):
        if visited[y0, x0]:
            continue
        blob = [(y0, x0)]
        visited[y0, x0] = True
        queue = deque(blob)
        while queue:
            y, x = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < m and 0 <= xx < n and BW[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        blob.append((yy, xx))
                        queue.append((yy, xx))
        if len(blob) > best_size:
            best, best_size = blob, len(blob)
    out = np.zeros_like(BW, dtype=bool)
    if best:
        ys, xs = zip(*best)
        out[list(ys), list(xs)] = True
    return out


blobs = {}
stages = {}                                   # masks of every stage, for the figures
for name, t in THRESHOLDS.items():
    BW_raw = threshold(t)                     # stage 1: thresholded
    BW_med = median_filter(BW_raw, MEDIAN_SIZE)   # stage 2: speckle removed
    BW_blob = largest_blob(BW_med)            # stage 3: one region per fruit
    stages[name] = (BW_raw, BW_med, BW_blob)
    blobs[name] = BW_blob

# %% Box & centroids
# Note: NumPy indexes arrays as [row, column] = [y, x], the same swap as in
# MATLAB. x runs left to right, y runs top to bottom.
print(f"darkFactor = {darkFactor}, mirror = {mirror}")
print(f"{'Fruit':<10}{'Box x: min-max':<16}{'Box y: min-max':<16}"
      f"{'a) Box midpoint':<20}{'b) Moment centroid':<22}{'Distance (px)'}")
results = {}
yy, xx = np.mgrid[0:m, 0:n]                          # y and x of every pixel
for name, BW in blobs.items():
    if not BW.any():
        print(f"{name:<10}not found")
        continue
    ys, xs = np.nonzero(BW)                          # like find() in MATLAB
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()

    # a) midpoint of the bounding box: average of max and min
    xa, ya = (xmin + xmax) / 2, (ymin + ymax) / 2

    # b) moment method: Mpq = sum over the image of x^p * y^q * I(x,y),
    #    with I(x,y) = 1 for blob pixels and 0 otherwise
    Ibin = BW.astype(float)
    M00 = np.sum(Ibin)                               # area (number of pixels)
    M10 = np.sum(xx * Ibin)
    M01 = np.sum(yy * Ibin)
    Xc, Yc = M10 / M00, M01 / M00

    dist = np.sqrt((xa - Xc) ** 2 + (ya - Yc) ** 2)
    results[name] = (xmin, xmax, ymin, ymax, xa, ya, Xc, Yc)
    print(f"{name:<10}{f'{xmin}-{xmax}':<16}{f'{ymin}-{ymax}':<16}"
          f"{f'({xa:.1f}, {ya:.1f})':<20}{f'({Xc:.1f}, {Yc:.1f})':<22}{dist:.2f}")

# %% Stage images
# Black-and-white pixel maps, one row per fruit: white = pixel counted as the
# fruit at that stage. Left to right: 1) after thresholding (speckle and stray
# patches are still there), 2) after the median filter (speckle gone, small
# holes closed), 3) after keeping the largest blob (one clean region -- this is
# what the box and centroids are computed from).
if SHOW_STAGES:
    fig2, axes = plt.subplots(3, 3, figsize=(12, 8.5))
    stage_titles = ["1) thresholded", "2) median filtered", "3) largest blob"]
    for row, (name, masks) in enumerate(stages.items()):
        for col, BW in enumerate(masks):
            axes[row, col].imshow(BW, cmap="gray")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(stage_titles[col])
        axes[row, 0].set_ylabel(name, fontsize=12)
    fig2.suptitle(f"Masks at each stage (darkFactor = {darkFactor}, mirror = {mirror})")
    plt.tight_layout()

    # The input before and after normalising: darkening dims the image, the
    # normalise step brings the brightness back (only the rounding remains).
    fig3, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    axL.imshow(I_show)
    axL.set_title(f"after darkening (darkFactor = {darkFactor})")
    axR.imshow(np.clip(np.round(I), 0, 255).astype(np.uint8))
    axR.set_title("after normalising (x 255 / max)")
    for a in (axL, axR):
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()

# %% Plot
fig, ax = plt.subplots(figsize=(11, 8.5))
ax.imshow(I_show)
for name, (xmin, xmax, ymin, ymax, xa, ya, Xc, Yc) in results.items():
    col = COLOURS[name]
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, edgecolor=col, linewidth=2.5))
    ax.plot(xa, ya, "+", color=col, markersize=16, markeredgewidth=3)
    ax.plot(Xc, Yc, "o", markerfacecolor="none", markeredgecolor=col,
            markersize=11, markeredgewidth=2.5)
    ax.text(xmin, ymin - 8, name, color=col, fontsize=13, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
ax.plot([], [], "w+", markersize=12, markeredgewidth=2, label="a) box midpoint")
ax.plot([], [], "wo", markerfacecolor="none", markersize=9, label="b) moment centroid")
ax.legend(loc="lower left", facecolor="gray")
ax.set_title(f"Task 1: fruits found (darkFactor = {darkFactor}, mirror = {mirror})")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
plt.tight_layout()
plt.show()