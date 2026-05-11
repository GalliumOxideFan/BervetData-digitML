import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from digitRecognition import whichNumber

# -----------------------
# Canvas
# -----------------------
grid = np.zeros((28, 28), dtype=float)
barData = []

fig, (ax_draw,ax_bar) = plt.subplots(1,2, figsize=(14,12), gridspec_kw={'width_ratios': [3,1]})

# leave space at bottom for button
plt.suptitle('Draw a digit, then press Calculate Digit to calculate what digit was drawn.', fontsize=25)
plt.subplots_adjust(bottom=0.2)

im = ax_draw.imshow(grid, cmap='gray', vmin=0, vmax=1)

ax_draw.set_xticks([])
ax_draw.set_yticks([])
ax_draw.set_frame_on(False)

# Bar plot
ax_bar.set_title('Certainty of digit', fontsize=20)
ax_bar.set_xlim(0,1)
ax_bar.set_yticks(range(10))
ax_bar.set_yticklabels([str(i) for i in range(10)])
ax_bar.set_xticks([])          # removes tick marks + labels

# -----------------------
# Update chart function
# -----------------------
def update_chart():
    ax_bar.clear()
    
    ax_bar.set_yticks(range(10))
    ax_bar.set_yticklabels([str(i) for i in range(10)])
    ax_bar.set_xticks([])          # removes tick marks + labels
    ax_bar.barh(range(len(barData)), barData)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_title("Certainty of digit", fontsize=20)
    fig.canvas.draw_idle()

# -----------------------
# Gaussian brush
# -----------------------
def make_brush(r, sigma=None):
    if sigma is None:
        sigma = r / 1.1

    size = 2 * r + 1
    k = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            x = i - r
            y = j - r
            k[i, j] = np.exp(-(x*x + y*y) / (1 * sigma * sigma))

    k /= k.max()
    return k

brush_radius = 2
brush = make_brush(brush_radius)

# -----------------------
# State
# -----------------------
drawing = False
last_pos = None

# -----------------------
# Paint system
# -----------------------
def stamp(x, y):
    r = brush_radius

    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            xi, yj = x + i, y + j

            if 0 <= xi < 28 and 0 <= yj < 28:
                val = brush[i + r, j + r]
                grid[yj, xi] = max(grid[yj, xi], val)

def draw_line(x0, y0, x1, y1):
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1

    for t in np.linspace(0, 1, steps):
        x = int(round(x0 + t * (x1 - x0)))
        y = int(round(y0 + t * (y1 - y0)))
        stamp(x, y)

# -----------------------
# Mouse events
# -----------------------
def on_press(event):
    global drawing, last_pos
    if event.inaxes != ax_draw:
        return
    drawing = True
    last_pos = None
    on_move(event)

def on_release(event):
    global drawing, last_pos
    drawing = False
    last_pos = None

def on_move(event):
    global last_pos

    if not drawing or event.inaxes != ax_draw:
        return
    if event.xdata is None or event.ydata is None:
        return

    x, y = int(event.xdata), int(event.ydata)

    if last_pos is None:
        stamp(x, y)
    else:
        draw_line(last_pos[0], last_pos[1], x, y)

    last_pos = (x, y)

    im.set_data(grid)
    fig.canvas.draw_idle()

# -----------------------
# Button (bottom)
# -----------------------
button_ax_draw = fig.add_axes([0.25, 0.05, 0.2, 0.075])
clear_button = Button(button_ax_draw, 'Clear')

def clear(event):
    global grid
    grid[:] = 0
    im.set_data(grid)
    fig.canvas.draw_idle()

clear_button.on_clicked(clear)

# -----------------------
# Second button
# -----------------------
button2_ax_draw = fig.add_axes([0.6, 0.05, 0.2, 0.075])
calc_button = Button(button2_ax_draw, 'Calculate digit')

def calculateDigit(event):
    global grid, barData
    griddy = np.flipud(np.rot90(grid))
    d = griddy.flatten()
    identifiedNumber, res = whichNumber(d)

    res = 1/res
    res = res/np.linalg.norm(res)
    res = (res/np.max(res))**5
    barData = res
    update_chart()
    ax_draw.set_title(f'Identified digit was {identifiedNumber}')
    fig.canvas.draw_idle()

calc_button.on_clicked(calculateDigit)

# -----------------------
# Connect events
# -----------------------
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()