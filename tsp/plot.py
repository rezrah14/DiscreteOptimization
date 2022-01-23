from typing import List
import matplotlib.pyplot as plt


class RoutePlot:
    def __init__(self, points):
        self.lines = None
        plt.ion()
        self.fig = plt.figure()
        self.points = points
        self.ax = self.fig.add_subplot()
        x_values = [p.x for p in points]
        y_values = [p.y for p in points]
        self.dots = self.ax.scatter(x_values, y_values, s=10)
        for i, p in enumerate(points):
            self.ax.text(p.x, p.y, "%d" % i, ha="center")

    def add_route(self, route, cost=None, color='red'):
        x_values = [self.points[i].x for i in route]
        y_values = [self.points[i].y for i in route]
        if cost is not None:
            self.ax.set_title(f"TWO-OPT Cost:{round(cost, 2)}", fontsize=16)
        self.lines, = self.ax.plot(x_values, y_values, linewidth=1, color=color)

    def update_data(self, route, cost, best_cost):
        x_values = [self.points[i].x for i in route]
        y_values = [self.points[i].y for i in route]
        if cost is not None:
            self.ax.set_title(f"TWO-OPT Cost:{round(cost, 2)}/{round(best_cost, 2)}", fontsize=16)
        self.lines.set_xdata(x_values)
        self.lines.set_ydata(y_values)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)


class CostPlot:
    def __init__(self, costs):
        plt.ion()
        self.fig = plt.figure()
        ax = self.fig.add_subplot()
        x_values = [x for x in range(len(costs))]
        y_values = [c for c in costs]
        self.lines,  = ax.plot(x_values, y_values, linewidth=1, color='blue')


def plot(x_values: List[int], y_values: List[int], idx=0):
    plt.figure()
    fig, ax = plt.subplots()
    ax.scatter(x_values, y_values, s=10)
    ax.plot(x_values, y_values, linewidth=1, color='red')
    # ax.set_xlabel("value", fontsize=14)
    # ax.set_ylabel("square", fontsize=14)
    # ax.tick_params(axis='both', labelsize=14)
    plt.savefig(f'img_{idx}.jpg')
    #plt.show()