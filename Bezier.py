import tkinter as tk
import numpy as np
from functools import partial

# Bézier and derivative functions
def bezier_curve(control_points, t):
    n = len(control_points) - 1
    curve_point = np.zeros(2)

    def bernstein_polynomial(n, i, t):
        return (np.math.comb(n, i)) * ((1 - t) ** (n - i)) * (t ** i)

    for i, p in enumerate(control_points):
        bernstein = bernstein_polynomial(n, i, t)
        curve_point += bernstein * np.array(p)
    return curve_point

def bezier_first_derivative(control_points, t):
    n = len(control_points) - 1
    derivative = np.zeros(2)
    for i in range(n):
        derivative += n * (control_points[i + 1] - control_points[i]) * ((1 - t) ** (n - 1 - i)) * (t ** i)
    return derivative

def bezier_second_derivative(control_points, t):
    n = len(control_points) - 1
    second_derivative = np.zeros(2)
    one_minus_t = (1 - t)
    t_power = 1
    for i in range(n - 1):
        second_derivative += n * (n - 1) * (control_points[i + 2] - 2 * control_points[i + 1] + control_points[i]) * (one_minus_t ** (n - 2 - i)) * t_power
        t_power *= t
    return second_derivative

def curvature_and_sign(control_points, t):
    d1 = bezier_first_derivative(control_points, t)
    d2 = bezier_second_derivative(control_points, t)
    d1_norm = np.linalg.norm(d1)
    cross_product = d1[0] * d2[1] - d1[1] * d2[0]
    curvature = abs(cross_product) / (d1_norm ** 3) if d1_norm != 0 else 0
    return curvature, np.sign(cross_product)

# GUI logic
def on_drag(index, event):
    control_points[index][0] = event.x
    control_points[index][1] = event.y
    canvas.coords(control_circles[index], event.x - 5, event.y - 5, event.x + 5, event.y + 5)
    canvas.coords(control_labels[index], event.x + 10, event.y)
    canvas.coords(coord_labels[index], event.x + 15, event.y + 15)
    canvas.itemconfig(coord_labels[index], text=f"({event.x:.1f}, {event.y:.1f})")
    if label_coords_visible:
        canvas.itemconfig(coord_labels[index], state='normal')
    draw_curve()
    update_orange_dot_and_circle()
    if show_lines:
        draw_straight_lines()

def update_orange_dot_and_circle():
    t = slider.get()
    orange_point = bezier_curve(control_points, t)
    canvas.coords(orange_dot, orange_point[0] - 3, orange_point[1] - 3, orange_point[0] + 3, orange_point[1] + 3)

    for i, weight_label in enumerate(weight_labels):
        bernstein = (np.math.comb(len(control_points) - 1, i)) * ((1 - t) ** (len(control_points) - 1 - i)) * (t ** i)
        weights[i] = bernstein
        weight_label.config(text=f"Weight {i + 1}: {weights[i]:.2f}")

    kappa, sign = curvature_and_sign(control_points, t)
    radius = float('inf') if kappa == 0 else round(1 / kappa, 10)
    d = 0.01
    point_next = bezier_curve(control_points, t + d)
    direction_vector = point_next - orange_point
    direction_vector /= np.linalg.norm(direction_vector)
    normal_vec = np.array([-direction_vector[1], direction_vector[0]])

    if radius > 0 and np.isfinite(radius):
        d2 = bezier_second_derivative(control_points, t)
        if np.dot(normal_vec, d2) < 0:
            normal_vec = -normal_vec
        circle_center = np.round(orange_point + radius * normal_vec, 10)

        if circle_visible:
            canvas.coords(orange_circle,
                          circle_center[0] - radius, circle_center[1] - radius,
                          circle_center[0] + radius, circle_center[1] + radius)
            canvas.coords(radius_line, orange_point[0], orange_point[1], circle_center[0], circle_center[1])
            midpoint_x = (orange_point[0] * 0.9 + circle_center[0] * 0.1)
            midpoint_y = (orange_point[1] * 0.9 + circle_center[1] * 0.1)
            canvas.coords(radius_label, midpoint_x, midpoint_y)
            canvas.itemconfig(radius_label, text=f"1/κ = 1/{kappa:.10f}")
        else:
            canvas.coords(orange_circle, 0, 0, 0, 0)
            canvas.coords(radius_line, 0, 0, 0, 0)
            canvas.itemconfig(radius_label, text="")
    else:
        canvas.coords(orange_circle, 0, 0, 0, 0)
        canvas.coords(radius_line, 0, 0, 0, 0)
        canvas.itemconfig(radius_label, text="")

    draw_tangent_line(t, orange_point)

def draw_tangent_line(t, point):
    d = 0.01
    point_next = bezier_curve(control_points, t + d)
    direction_vector = point_next - point
    direction_vector /= np.linalg.norm(direction_vector)
    tangent_start = point - 50 * direction_vector
    tangent_end = point + 50 * direction_vector
    canvas.coords(tangent_line, tangent_start[0], tangent_start[1], tangent_end[0], tangent_end[1])

def draw_curve():
    if not curve_visible:
        canvas.delete("curve")
        return
    draw_t_dots()
    canvas.delete("curve")
    t_values = np.linspace(0, 1, 100)
    bezier_points = [bezier_curve(control_points, t) for t in t_values]
    curve_coords = [coord for point in bezier_points for coord in point]
    canvas.create_line(curve_coords, fill="blue", width=2, tags="curve")

def add_control_point():
    global weights
    new_point = np.array([100, 100])
    control_points.append(new_point)
    weights = np.append(weights, 1.0)
    circle = canvas.create_oval(new_point[0] - 5, new_point[1] - 5, new_point[0] + 5, new_point[1] + 5, fill='red')
    control_circles.append(circle)
    label = canvas.create_text(new_point[0] + 10, new_point[1], text=str(len(control_points)), fill="black", font=("Arial", 12))
    control_labels.append(label)
    weight_label = tk.Label(weights_frame, text=f"Weight {len(control_points)}: 1.00")
    weight_label.pack(pady=2)
    weight_labels.append(weight_label)
    canvas.tag_bind(circle, "<B1-Motion>", partial(on_drag, len(control_points) - 1))
    coord_label = canvas.create_text(new_point[0] + 15, new_point[1] + 15, text=f"({new_point[0]:.1f}, {new_point[1]:.1f})",
                                     fill="blue", font=("Arial", 10), state='hidden')
    coord_labels.append(coord_label)
    draw_curve()
    update_orange_dot_and_circle()
    if show_lines:
        draw_straight_lines()

def remove_control_point():
    global weights
    if len(control_points) > 2:
        canvas.delete(control_circles.pop())
        canvas.delete(control_labels.pop())
        canvas.delete(coord_labels.pop())
        control_points.pop()
        weights = weights[:-1]
        weight_labels[-1].destroy()
        weight_labels.pop()
        draw_curve()
        update_orange_dot_and_circle()
        if show_lines:
            draw_straight_lines()

def draw_straight_lines():
    canvas.delete("lines")
    for i in range(len(control_points) - 1):
        canvas.create_line(control_points[i][0], control_points[i][1],
                           control_points[i + 1][0], control_points[i + 1][1],
                           fill="red", tags="lines")

def toggle_lines():
    global show_lines
    show_lines = not show_lines
    if show_lines:
        draw_straight_lines()
    else:
        canvas.delete("lines")

def toggle_circle():
    global circle_visible
    circle_visible = not circle_visible
    update_orange_dot_and_circle()

def toggle_curve():
    global curve_visible
    curve_visible = not curve_visible
    draw_curve()

def toggle_t_dots():
    global t_dots_visible
    t_dots_visible = not t_dots_visible
    draw_t_dots()

def draw_t_dots():
    canvas.delete("t_dot")
    if t_dots_visible:
        num_dots = t_dots_count.get()
        t_values = np.linspace(0, 1, num_dots)
        for t in t_values:
            point = bezier_curve(control_points, t)
            canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3,
                               fill="light green", tags="t_dot")

def toggle_label_coords():
    global label_coords_visible
    label_coords_visible = not label_coords_visible
    for label in coord_labels:
        canvas.itemconfig(label, state='normal' if label_coords_visible else 'hidden')

# Init
control_points = [np.array([50, 400]), np.array([150, 200]), np.array([300, 300]), np.array([400, 100])]
weights = np.ones(len(control_points))

window = tk.Tk()
window.title("Interactive Bézier Curve with Osculating Circle")

# GUI layout
main_frame = tk.Frame(window)
main_frame.pack()
weights_frame = tk.Frame(main_frame)
weights_frame.pack(side=tk.LEFT, padx=10)
canvas_width = 1000
canvas_height = 800
canvas = tk.Canvas(main_frame, width=canvas_width, height=canvas_height, bg="white")
canvas.pack(side=tk.LEFT)
control_frame = tk.Frame(main_frame)
control_frame.pack(side=tk.RIGHT, padx=10)

# Buttons
tk.Button(control_frame, text="Add Point", command=add_control_point).pack(pady=5)
tk.Button(control_frame, text="Remove Point", command=remove_control_point).pack(pady=5)
tk.Button(control_frame, text="Toggle Lines", command=toggle_lines).pack(pady=5)
tk.Button(control_frame, text="Toggle Circle", command=toggle_circle).pack(pady=5)
tk.Button(control_frame, text="Toggle Curve", command=toggle_curve).pack(pady=5)
tk.Button(control_frame, text="Toggle t Dots", command=toggle_t_dots).pack(pady=5)
tk.Button(control_frame, text="Toggle Coordinate Labels", command=toggle_label_coords).pack(pady=5)

tk.Label(control_frame, text="Number of t Dots:").pack(pady=2)
t_dots_count = tk.IntVar(value=20)
t_dots_count.trace_add("write", lambda *args: draw_t_dots())
tk.OptionMenu(control_frame, t_dots_count, 20, 30, 50).pack(pady=5)

slider_frame = tk.Frame(window)
slider_frame.pack()
tk.Label(slider_frame, text="t-value:").pack(side=tk.LEFT)
slider = tk.Scale(slider_frame, from_=0, to=1, orient=tk.HORIZONTAL, resolution=0.01, command=lambda x: update_orange_dot_and_circle())
slider.pack(side=tk.LEFT)

# Initialize visuals
circle_visible = True
show_lines = False
t_dots_visible = False
label_coords_visible = False
curve_visible = True

weight_labels = []
for i in range(len(control_points)):
    label = tk.Label(weights_frame, text=f"Weight {i + 1}: {weights[i]:.2f}")
    label.pack(pady=2)
    weight_labels.append(label)

control_circles = []
control_labels = []
coord_labels = []
for i, point in enumerate(control_points):
    circle = canvas.create_oval(point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5, fill='red')
    control_circles.append(circle)
    label = canvas.create_text(point[0] + 10, point[1], text=str(i + 1), fill="black", font=("Arial", 12))
    control_labels.append(label)
    canvas.tag_bind(circle, "<B1-Motion>", partial(on_drag, i))
    coord_label = canvas.create_text(point[0] + 15, point[1] + 15, text=f"({point[0]:.1f}, {point[1]:.1f})",
                                     fill="blue", font=("Arial", 10), state='hidden')
    coord_labels.append(coord_label)

draw_curve()
initial_point = bezier_curve(control_points, 0)
orange_dot = canvas.create_oval(initial_point[0] - 3, initial_point[1] - 3, initial_point[0] + 3, initial_point[1] + 3, outline="orange", width=2)
orange_circle = canvas.create_oval(0, 0, 0, 0, outline="orange")
radius_line = canvas.create_line(0, 0, 0, 0, fill="orange", dash=(4, 2))
radius_label = canvas.create_text(0, 0, text="", fill="orange", font=("Arial", 10))
tangent_line = canvas.create_line(0, 0, 0, 0, fill="green", dash=(4, 2))

window.mainloop()
