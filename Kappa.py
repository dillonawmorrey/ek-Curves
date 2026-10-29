import tkinter as tk
import numpy as np
from scipy.optimize import root_scalar
from scipy.linalg import solve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Global variables
global lambda_values, t_values, Q_i_0_set, Q_i_1_set, Q_i_2_set, q_points_visible
q_points_visible = True

# Constants for control point display and hit area
CONTROL_RADIUS = 5  # visible circle radius
HIT_RADIUS = 20     # invisible hit area radius

def bezier_curve(control_points, t):
    n = len(control_points) - 1
    bernstein_polynomials = [
        (np.math.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i))
        for i in range(n + 1)
    ]
    return np.sum([bernstein * np.array(p[:2])
                   for bernstein, p in zip(bernstein_polynomials, control_points)], axis=0)

def initialize_Q_points(control_points):
    n = len(control_points) - 2
    Q_i_0_set = np.zeros((n, 2))
    Q_i_1_set = np.zeros((n, 2))
    Q_i_2_set = np.zeros((n, 2))
    for j in range(1, n + 1):
        Q_i_1_set[j - 1] = control_points[j]
    Q_i_0_set[0] = control_points[0]
    Q_i_2_set[-1] = control_points[-1]
    return Q_i_0_set, Q_i_1_set, Q_i_2_set

def update_Q_i_0_Q_i_2(Q_i_0_set, Q_i_1_set, Q_i_2_set, lambda_values):
    n = len(Q_i_1_set)
    for i in range(n):
        if i != n - 1:
            Q_i_2_set[i] = (1 - lambda_values[i]) * Q_i_1_set[i] + lambda_values[i] * Q_i_1_set[i + 1]
        if i != 0:
            Q_i_0_set[i] = Q_i_2_set[i - 1]

def t_i_update(control_points, t_values, Q_i_0_set, Q_i_2_set):
    n = len(control_points) - 2
    for i in range(n):
        Q_i_0 = Q_i_0_set[i]
        Q_i_2 = Q_i_2_set[i]
        P_i = control_points[i + 1]

        def cubic_equation(t):
            term1 = ((np.linalg.norm(Q_i_2 - Q_i_0)) ** 2) * (t**3)
            term2 = (3 * np.dot(Q_i_2 - Q_i_0, Q_i_0 - P_i)) * (t**2)
            term3 = (np.dot((3 * Q_i_0) - (2 * P_i) - Q_i_2, (Q_i_0 - P_i) * t))
            term4 = -(np.linalg.norm(Q_i_0 - P_i) ** 2)
            return term1 + term2 + term3 + term4

        t_initial_guess = t_values[i]
        result = root_scalar(cubic_equation, bracket=[0, 1], method='brentq')
        t_values[i] = round(result.root if result.converged else t_initial_guess, 8)
        if t_values[i] == 0:
            t_values[i] = 0.5

    return t_values

def lambda_i_update(Q_i_0_set, Q_i_1_set, Q_i_2_set, lambda_values):
    delta = lambda Q1, Q2, Q3: 0.5 * abs(
        Q1[0] * Q2[1] - Q2[0] * Q1[1] +
        Q2[0] * Q3[1] - Q3[0] * Q2[1] +
        Q3[0] * Q1[1] - Q1[0] * Q3[1]
    )
    deltas = [
        delta(Q_i_0_set[i - 1], Q_i_1_set[i - 1], Q_i_1_set[i])
        for i in range(1, len(Q_i_1_set))
    ]
    numerator = np.sqrt(deltas)
    denominator = numerator + np.sqrt([
        delta(Q_i_1_set[i - 1], Q_i_1_set[i], Q_i_2_set[i])
        for i in range(1, len(Q_i_1_set))
    ])
    lambda_values[:] = np.clip(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0), 0, 1)
    return lambda_values

def Q_i_j_update(control_points, t_values, lambda_values, Q_i_1_set):
    n = len(control_points) - 2
    A = lil_matrix((n, n))
    B = np.zeros((n, 2))

    for i in range(n):
        if i == 0:
            A[i, i] = 2 * (1 - t_values[i]) * t_values[i] + (1 - lambda_values[i]) * (t_values[i] ** 2)
            A[i, i + 1] = lambda_values[i] * (t_values[i] ** 2)
            B[i] = control_points[i + 1] - ((1 - t_values[i]) ** 2 * control_points[i])
        elif i == n - 1:
            A[i, i - 1] = (1 - lambda_values[i - 1]) * ((1 - t_values[i]) ** 2)
            A[i, i] = lambda_values[i - 1] * ((1 - t_values[i]) ** 2) + 2 * ((1 - t_values[i]) * t_values[i])
            B[i] = control_points[i + 1] - (t_values[i] ** 2 * control_points[i + 2])
        else:
            A[i, i - 1] = (1 - lambda_values[i - 1]) * ((1 - t_values[i]) ** 2)
            A[i, i] = (lambda_values[i - 1] * ((1 - t_values[i]) ** 2) +
                       2 * ((1 - t_values[i]) * t_values[i]) +
                       ((1 - lambda_values[i]) * (t_values[i] ** 2)))
            A[i, i + 1] = lambda_values[i] * (t_values[i] ** 2)
            B[i] = control_points[i + 1]

    Q_i_1_solution = spsolve(A.tocsr(), B)
    Q_i_1_set[:n] = Q_i_1_solution
    return Q_i_1_set

def run_kappa_curve_optimization(control_points, iterations=60):
    t_values = np.full(len(control_points) - 2, 0.5)
    lambda_values = np.full(len(control_points) - 3, 0.5)
    Q_i_0_set, Q_i_1_set, Q_i_2_set = initialize_Q_points(control_points)

    for _ in range(iterations):
        lambda_values = lambda_i_update(Q_i_0_set, Q_i_1_set, Q_i_2_set, lambda_values)
        update_Q_i_0_Q_i_2(Q_i_0_set, Q_i_1_set, Q_i_2_set, lambda_values)
        t_values = t_i_update(control_points, t_values, Q_i_0_set, Q_i_2_set)
        Q_i_1_set = Q_i_j_update(control_points, t_values, lambda_values, Q_i_1_set)

    return Q_i_0_set, Q_i_1_set, Q_i_2_set

def toggle_lines():
    global show_lines
    show_lines = not show_lines
    draw_curve()

def toggle_label_coords():
    global label_coords_visible
    label_coords_visible = not label_coords_visible
    for coord_label in coord_labels:
        canvas.itemconfig(coord_label, state='normal' if label_coords_visible else 'hidden')

def toggle_q_points():
    global q_points_visible, Q_i_0_set, Q_i_1_set, Q_i_2_set
    q_points_visible = not q_points_visible
    # Recalculate the Q points using the current control points and iterations
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())
    draw_q_points(Q_i_0_set, Q_i_1_set, Q_i_2_set)

def on_drag(event, index):
    # Update the control point position
    control_points[index] = [event.x, event.y]
    # Update the visible circle
    canvas.coords(control_circles[index],
                  event.x - CONTROL_RADIUS, event.y - CONTROL_RADIUS,
                  event.x + CONTROL_RADIUS, event.y + CONTROL_RADIUS)
    # Update the invisible hit area
    canvas.coords(control_hit_areas[index],
                  event.x - HIT_RADIUS, event.y - HIT_RADIUS,
                  event.x + HIT_RADIUS, event.y + HIT_RADIUS)
    # Update the label position
    canvas.coords(control_labels[index], event.x + 10, event.y)
    # Update the coordinate label
    canvas.coords(coord_labels[index], event.x + 15, event.y + 15)
    canvas.itemconfig(coord_labels[index], text=f"({event.x:.8f}, {event.y:.8f})")
    # Redraw the curve and update Q points
    draw_curve()
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())
    draw_q_points(Q_i_0_set, Q_i_1_set, Q_i_2_set)

def draw_curve():
    canvas.delete("curve")
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())

    if show_lines:
        for i in range(len(Q_i_0_set)):
            canvas.create_line(Q_i_0_set[i][0], Q_i_0_set[i][1],
                               Q_i_1_set[i][0], Q_i_1_set[i][1],
                               fill="gray", tags="curve")
            canvas.create_line(Q_i_1_set[i][0], Q_i_1_set[i][1],
                               Q_i_2_set[i][0], Q_i_2_set[i][1],
                               fill="gray", tags="curve")

    for i in range(len(Q_i_0_set)):
        final_control_points = np.vstack((Q_i_0_set[i], Q_i_1_set[i], Q_i_2_set[i]))
        t_vals = np.linspace(0, 1, 100)
        bezier_points = [bezier_curve(final_control_points, t) for t in t_vals]
        curve_coords = [coord for point in bezier_points for coord in point]
        canvas.create_line(curve_coords, fill="blue", width=2, tags="curve")
        
    # Raise the invisible hit areas above the curve so they can capture drag events
    for hit in control_hit_areas:
        canvas.tag_raise(hit)

def draw_q_points(Q_i_0_set, Q_i_1_set, Q_i_2_set):
    canvas.delete("q_points")
    if not q_points_visible:
        return
    for i in range(len(Q_i_1_set)):
        q_points = [Q_i_0_set[i], Q_i_1_set[i], Q_i_2_set[i]]
        for j, q in enumerate(q_points):
            q_label = f"Q{i+1},{j}"
            canvas.create_rectangle(q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5,
                                    outline='black', fill='', tags="q_points")
            canvas.create_text(q[0] + 10, q[1] - 10, text=q_label,
                               fill="black", font=("Arial", 10), tags="q_points")


def add_control_point():
    global control_points, control_circles, control_hit_areas, control_labels, coord_labels
    new_point = np.array([canvas_width / 2, canvas_height / 2])
    control_points = np.vstack((control_points, new_point))
    

    hit_area = canvas.create_oval(new_point[0] - HIT_RADIUS, new_point[1] - HIT_RADIUS,
                                  new_point[0] + HIT_RADIUS, new_point[1] + HIT_RADIUS,
                                  fill="", outline="")
    control_hit_areas.append(hit_area)
  
    circle = canvas.create_oval(new_point[0] - CONTROL_RADIUS, new_point[1] - CONTROL_RADIUS,
                                new_point[0] + CONTROL_RADIUS, new_point[1] + CONTROL_RADIUS,
                                fill='red')
    control_circles.append(circle)
    # Create label for the control point
    new_index = len(control_points)  
    label = canvas.create_text(new_point[0] + 10, new_point[1],
                               text=str(new_index), fill="black", font=("Arial", 12))
    control_labels.append(label)
    # Create coordinate label
    coord_label = canvas.create_text(new_point[0] + 15, new_point[1] + 15,
                                     text=f"({new_point[0]:.8f}, {new_point[1]:.8f})",
                                     fill="blue", font=("Arial", 10),
                                     state='normal' if label_coords_visible else 'hidden')
    coord_labels.append(coord_label)
    
 
    canvas.tag_bind(hit_area, "<B1-Motion>", lambda event, idx=len(control_points)-1: on_drag(event, idx))
    canvas.tag_bind(circle, "<B1-Motion>", lambda event, idx=len(control_points)-1: on_drag(event, idx))
    
    draw_curve()

def remove_control_point():
    global control_points, control_circles, control_hit_areas, control_labels, coord_labels
    if len(control_points) > 5:
        index = len(control_points) - 1 
        canvas.delete(control_hit_areas[index])
        canvas.delete(control_circles[index])
        canvas.delete(control_labels[index])
        canvas.delete(coord_labels[index])
        control_hit_areas.pop()
        control_circles.pop()
        control_labels.pop()
        coord_labels.pop()
        control_points = np.delete(control_points, index, axis=0)
        draw_curve()
        message_label.config(text="")
    else:
        message_label.config(text="Minimum of 5 control points required.")
        window.after(5000, lambda: message_label.config(text=""))



###### Tkinter Setup #######
window = tk.Tk()
window.title("Interactive κ-Curve")
label_coords_visible = False
show_lines = False
canvas_width = 800
canvas_height = 600
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

iterations_var = tk.IntVar(value=60)
iterations_spinbox = tk.Spinbox(window, from_=1, to=1000, increment=1,
                                textvariable=iterations_var, command=draw_curve, width=5)
iterations_spinbox.pack(pady=5)

toggle_lines_button = tk.Button(window, text="Toggle Lines", command=toggle_lines)
toggle_lines_button.pack(pady=5)

toggle_label_coords_button = tk.Button(window, text="Toggle Coordinate Labels", command=toggle_label_coords)
toggle_label_coords_button.pack(pady=5)

toggle_q_points_button = tk.Button(window, text="Toggle Q Points", command=toggle_q_points)
toggle_q_points_button.pack(pady=5)


add_cp_button = tk.Button(window, text="Add Control Point", command=add_control_point)
add_cp_button.pack(pady=5)

remove_cp_button = tk.Button(window, text="Remove Control Point", command=remove_control_point)
remove_cp_button.pack(pady=5)


message_label = tk.Label(window, text="", fg="red")
message_label.pack(pady=5)

# Initialize control points and associated canvas items
control_points = np.array([[50, 400], [250, 200], [350, 300], [450, 400], [550, 350]])
control_hit_areas = []
control_circles = []
control_labels = []
coord_labels = []
for i, point in enumerate(control_points):
    hit_area = canvas.create_oval(point[0] - HIT_RADIUS, point[1] - HIT_RADIUS,
                                  point[0] + HIT_RADIUS, point[1] + HIT_RADIUS,
                                  fill="", outline="")
    control_hit_areas.append(hit_area)

    circle = canvas.create_oval(point[0] - CONTROL_RADIUS, point[1] - CONTROL_RADIUS,
                                point[0] + CONTROL_RADIUS, point[1] + CONTROL_RADIUS,
                                fill='red')
    control_circles.append(circle)
    label = canvas.create_text(point[0] + 10, point[1],
                               text=str(i + 1), fill="black", font=("Arial", 12))
    control_labels.append(label)

    canvas.tag_bind(hit_area, "<B1-Motion>", lambda event, idx=i: on_drag(event, idx))
    canvas.tag_bind(circle, "<B1-Motion>", lambda event, idx=i: on_drag(event, idx))
    coord_label = canvas.create_text(point[0] + 15, point[1] + 15,
                                     text=f"({point[0]:.8f}, {point[1]:.8f})",
                                     fill="blue", font=("Arial", 10),
                                     state='hidden')
    coord_labels.append(coord_label)

draw_curve()
Q_i_0, Q_i_1, Q_i_2 = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())
draw_q_points(Q_i_0, Q_i_1, Q_i_2)
window.mainloop()
