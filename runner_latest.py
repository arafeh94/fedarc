import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

colors = ['black', 'blue', 'red', 'yellow']


def add_line(p1, p2, color='blue', ax=plt):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=0.3)


pts = np.array([
    [1, 1],
    [4, 3],
    [7, 8],
    [3, 5]
])
plt.scatter(pts[:, 0], pts[:, 1], color='blue')
plt.show()

gm = [np.average(pts[:, 0]), np.average(pts[:, 1])]

plt.scatter(pts[:, 0], pts[:, 1], color='blue')
plt.scatter([gm[0]], [gm[1]], color='red')
for index, pt in enumerate(pts):
    add_line(pt, gm, 'black')

plt.show()

pts = np.append(pts, [
    [2, 5],
    [8, 6],
    [5, 2],
    [2, 0],
    [0, 10],
], axis=0)

plt.scatter(pts[:, 0], pts[:, 1], color='blue')
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)


def animate(round):
    ax1.clear()
    selected_idx = np.random.choice(len(pts), 5)
    selected = np.array([pts[i] for i in selected_idx])
    gm = [np.average(selected[:, 0]), np.average(selected[:, 1])]
    ax1.scatter(pts[:, 0], pts[:, 1], color='blue')
    ax1.scatter([gm[0]], [gm[1]], color='red')
    for index, pt in enumerate(selected):
        add_line(pt, gm, 'black', ax=ax1)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
