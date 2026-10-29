import tkinter as tk
import numpy as np
from scipy.optimize import brentq
import math

CONTROL_RADIUS = 5
HIT_RADIUS = 15
q_points_visible = True
label_coords_visible = False
show_lines = False




# Cubic polynomial from the paper 
''' 
Miura, K.T., Gobithaasan, R.U., Salvi, P. et al. 
ek-Curves: controlled local curvature extrema. Vis Comput 38, 2723–2738 (2022). https://doi.org/10.1007/s00371-021-02149-8
'''
def polynomial_cubic(t, x0, y0, x2, y2, px, py, a):
    coeffs = [
        -3*a*y0**2 + 4*y0**2 + 6*a*py*y0 - 8*py*y0 - 3*a*x0**2 + 4*x0**2
            + 6*a*px*x0 - 8*px*x0 - 3*a*py**2 + 4*py**2 - 3*a*px**2 + 4*px**2,

        18*a**2*y0*y2 - 36*a*y0*y2 + 18*y0*y2 - 18*a**2*py*y2 + 36*a*py*y2 - 18*py*y2
            -18*a**2*y0**2 + 48*a*y0**2 - 30*y0**2 + 18*a**2*py*y0 - 60*a*py*y0 + 42*py*y0
            +18*a**2*x0*x2 - 36*a*x0*x2 + 18*x0*x2 - 18*a**2*px*x2 + 36*a*px*x2 - 18*px*x2
            -18*a**2*x0**2 + 48*a*x0**2 - 30*x0**2 + 18*a**2*px*x0 - 60*a*px*x0 + 42*px*x0
            +12*a*py**2 - 12*py**2 + 12*a*px**2 - 12*px**2,

        (-144*a**2*y0*y2) + 258*a*y0*y2 - 114*y0*y2 + 144*a**2*py*y2 - 258*a*py*y2 + 114*py*y2
            +144*a**2*y0**2 - 276*a*y0**2 + 126*y0**2 -144*a**2*py*y0 + 294*a*py*y0 -138*py*y0
            -144*a**2*x0*x2 + 258*a*x0*x2 -114*x0*x2 +144*a**2*px*x2 - 258*a*px*x2 +114*px*x2
            +144*a**2*x0**2 -276*a*x0**2 +126*x0**2 -144*a**2*px*x0 +294*a*px*x0 -138*px*x0
            -18*a*py**2 +12*py**2 -18*a*px**2 +12*px**2,

        54*a**3*y2**2 -162*a**2*y2**2 +162*a*y2**2 -54*y2**2 -108*a**3*y0*y2
            +774*a**2*y0*y2 -1032*a*y0*y2 +380*y0*y2 -450*a**2*py*y2 +708*a*py*y2 -272*py*y2
            +54*a**3*y0**2 -612*a**2*y0**2 +882*a*y0**2 -334*y0**2 +450*a**2*py*y0
            -732*a*py*y0 +288*py*y0 +54*a**3*x2**2 -162*a**2*x2**2 +162*a*x2**2
            -54*x2**2 -108*a**3*x0*x2 +774*a**2*x0*x2 -1032*a*x0*x2 +380*x0*x2
            -450*a**2*px*x2 +708*a*px*x2 -272*px*x2 +54*a**3*x0**2 -612*a**2*x0**2
            +882*a*x0**2 -334*x0**2 +450*a**2*px*x0 -732*a*px*x0 +288*px*x0
            +12*a*py**2 -8*py**2 +12*a*px**2 -8*px**2,

        (-405*a**3*y2**2)+1080*a**2*y2**2 -945*a*y2**2 +270*y2**2 +810*a**3*y0*y2
            -2880*a**2*y0*y2 +2910*a*y0*y2 -900*y0*y2 +720*a**2*py*y2 -1020*a*py*y2
            +360*py*y2 -405*a**3*y0**2 +1800*a**2*y0**2 -1965*a*y0**2 +630*y0**2
            -720*a**2*py*y0 +1020*a*py*y0 -360*py*y0 -405*a**3*x2**2 +1080*a**2*x2**2
            -945*a*x2**2 +270*x2**2 +810*a**3*x0*x2 -2880*a**2*x0*x2 +2910*a*x0*x2
            -900*x0*x2 +720*a**2*px*x2 -1020*a*px*x2 +360*px*x2 -405*a**3*x0**2
            +1800*a**2*x0**2 -1965*a*x0**2 +630*x0**2 -720*a**2*px*x0 +1020*a*px*x0
            -360*px*x0,

        1296*a**3*y2**2 -3114*a**2*y2**2 +2454*a*y2**2 -636*y2**2 -2592*a**3*y0*y2
            +6822*a**2*y0*y2 -5700*a*y0*y2 +1536*y0*y2 -594*a**2*py*y2 +792*a*py*y2
            -264*py*y2 +1296*a**3*y0**2 -3708*a**2*y0**2 +3246*a*y0**2 -900*y0**2
            +594*a**2*py*y0 -792*a*py*y0 +264*py*y0 +1296*a**3*x2**2 -3114*a**2*x2**2
            +2454*a*x2**2 -636*x2**2 -2592*a**3*x0*x2 +6822*a**2*x0*x2 -5700*a*x0*x2
            +1536*x0*x2 -594*a**2*px*x2 +792*a*px*x2 -264*px*x2 +1296*a**3*x0**2
            -3708*a**2*x0**2 +3246*a*x0**2 -900*x0**2 +594*a**2*px*x0 -792*a*px*x0
            +264*px*x0,

        (-2268*a**3*y2**2)+5004*a**2*y2**2 -3648*a*y2**2 +880*y2**2 +4536*a**3*y0*y2
            -10206*a**2*y0*y2 +7560*a*y0*y2 -1848*y0*y2 +198*a**2*py*y2 -264*a*py*y2
            +88*py*y2 -2268*a**3*y0**2 +5202*a**2*y0**2 -3912*a*y0**2 +968*y0**2
            -198*a**2*py*y0 +264*a*py*y0 -88*py*y0 -2268*a**3*x2**2 +5004*a**2*x2**2
            -3648*a*x2**2 +880*x2**2 +4536*a**3*x0*x2 -10206*a**2*x0*x2 +7560*a*x0*x2
            -1848*x0*x2 +198*a**2*px*x2 -264*a*px*x2 +88*px*x2 -2268*a**3*x0**2
            +5202*a**2*x0**2 -3912*a*x0**2 +968*x0**2 -198*a**2*px*x0 +264*a*px*x0
            -88*px*x0,

        2268*a**3*y2**2 -4698*a**2*y2**2 +3240*a*y2**2 -744*y2**2 -4536*a**3*y0*y2
            +9396*a**2*y0*y2 -6480*a*y0*y2 +1488*y0*y2 +2268*a**3*y0**2 -4698*a**2*y0**2
            +3240*a*y0**2 -744*y0**2 +2268*a**3*x2**2 -4698*a**2*x2**2 +3240*a*x2**2
            -744*x2**2 -4536*a**3*x0*x2 +9396*a**2*x0*x2 -6480*a*x0*x2 +1488*x0*x2
            +2268*a**3*x0**2 -4698*a**2*x0**2 +3240*a*x0**2 -744*x0**2,

        (-1215*a**3*y2**2)+2430*a**2*y2**2 -1620*a*y2**2 +360*y2**2 +2430*a**3*y0*y2
            -4860*a**2*y0*y2 +3240*a*y0*y2 -720*y0*y2 -1215*a**3*y0**2 +2430*a**2*y0**2
            -1620*a*y0**2 +360*y0**2 -1215*a**3*x2**2 +2430*a**2*x2**2 -1620*a*x2**2
            +360*x2**2 +2430*a**3*x0*x2 -4860*a**2*x0*x2 +3240*a*x0*x2 -720*x0*x2
            -1215*a**3*x0**2 +2430*a**2*x0**2 -1620*a*x0**2 +360*x0**2,

        270*a**3*y2**2 -540*a**2*y2**2 +360*a*y2**2 -80*y2**2 -540*a**3*y0*y2
            +1080*a**2*y0*y2 -720*a*y0*y2 +160*y0*y2 +270*a**3*y0**2 -540*a**2*y0**2
            +360*a*y0**2 -80*y0**2 +270*a**3*x2**2 -540*a**2*x2**2 +360*a*x2**2
            -80*x2**2 -540*a**3*x0*x2 +1080*a**2*x0*x2 -720*a*x0*x2 +160*x0*x2
            +270*a**3*x0**2 -540*a**2*x0**2 +360*a*x0**2 -80*x0**2
    ]
    val = 0.0
    for i, c in enumerate(coeffs):
        val += c * (t**i)
    return val

def find_root_cubic(x0, y0, x2, y2, px, py, alpha):
    def f(t):
        return polynomial_cubic(t, x0, y0, x2, y2, px, py, alpha)
    try:
        if f(0)*f(1) > 0:
            return 0.5
        return brentq(f, 0, 1)
    except:
        return 0.5

def create_cubic(Q0, Q1, Q2, alpha):
    P0 = Q0
    P3 = Q2
    P1 = (1 - alpha)*Q0 + alpha*Q1
    P2 = (1 - alpha)*Q2 + alpha*Q1
    return np.array([P0, P1, P2, P3])

def bezier_cubic(cpoints, t):
    B0 = (1 - t)**3
    B1 = 3*(1 - t)**2 * t
    B2 = 3*(1 - t)*t**2
    B3 = t**3
    return B0*cpoints[0] + B1*cpoints[1] + B2*cpoints[2] + B3*cpoints[3]


# Q-points, area-based λ, etc.

def initialize_Q_points(control_points):
    n = len(control_points) - 2
    Q_i_0_set = np.zeros((n, 2))
    Q_i_1_set = np.zeros((n, 2))
    Q_i_2_set = np.zeros((n, 2))
    for j in range(n):
        Q_i_1_set[j] = control_points[j + 1]
    if n > 0:
        Q_i_0_set[0]  = control_points[0]
        Q_i_2_set[-1] = control_points[-1]
    return Q_i_0_set, Q_i_1_set, Q_i_2_set

def delta(a, b, c):
    return 0.5 * abs(
        a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1])
    )

def lambda_i_update(Q_i_0_set, Q_i_1_set, Q_i_2_set,a):
    epsilon = 10^(-10)
    n = len(Q_i_1_set)
    lam = np.zeros(n - 1) if n > 1 else np.array([])
    for i in range(n - 1):
        tmp = np.sqrt((1-a[i])*delta(Q_i_0_set[i], Q_i_1_set[i], Q_i_1_set[i+1])+epsilon)
        denom = tmp + (a[i]/a[i+1])*np.sqrt((1-a[i+1])*delta(Q_i_1_set[i], Q_i_1_set[i+1], Q_i_2_set[i+1])+epsilon)
        if denom < 1e-10:
            denom += 1e-10
        lam[i] = tmp / denom
    return lam

def t_i_update_cubic(Q_i_0_set, Q_i_1_set, Q_i_2_set, alpha_array):
    n = len(Q_i_1_set)
    t_values = []
    for i in range(n):
        x0, y0 = Q_i_0_set[i]
        x2, y2 = Q_i_2_set[i]
        px, py = Q_i_1_set[i]
        a_i = alpha_array[i]
        sol = find_root_cubic(x0, y0, x2, y2, px, py, a_i)
        t_values.append(sol)
    return np.array(t_values)

def update_Q_i_0_Q_i_2(Q_i_0_set, Q_i_1_set, Q_i_2_set, lam):
    n = len(Q_i_1_set)
    for i in range(n):
        if i < n - 1:
            Q_i_2_set[i] = (1 - lam[i]) * Q_i_1_set[i] + lam[i] * Q_i_1_set[i + 1]
        if i > 0:
            Q_i_0_set[i] = Q_i_2_set[i - 1]

def compute_central_cps_cubic(Q_i_0_set, Q_i_1_set, Q_i_2_set,
                              lam, t, interior_points, alpha_array):
    n = len(Q_i_1_set)
    A = np.zeros((n, n))
    fixed = np.zeros((n, 2))

    def left_basis(u, a):
        return (1 - u)**3 + 3*(1 - u)**2 * u * (1 - a)
    def right_basis(u, a):
        return u**3 + 3*(1 - u)*u**2 * (1 - a)
    def middle_basis(u, a):
        return a * 3*(1 - u)*u

    for i in range(n):
        u = t[i]
        a_i = alpha_array[i]

        if i > 0:
            A[i, i-1] = (1 - lam[i-1]) * left_basis(u, a_i)
        else:
            Q0_0 = Q_i_0_set[0]
            fixed[i] -= Q0_0 * left_basis(u, a_i)

        if i < n - 1:
            A[i, i+1] = lam[i] * right_basis(u, a_i)
        else:
            Qn_2 = Q_i_2_set[n - 1]
            fixed[i] -= Qn_2 * right_basis(u, a_i)

        if n == 1:
            A[i, i] = middle_basis(u, a_i)
        elif i == 0:
            A[i, i] = middle_basis(u, a_i) + (1 - lam[i]) * right_basis(u, a_i)
        elif i == n - 1:
            A[i, i] = lam[i - 1]*left_basis(u, a_i) + middle_basis(u, a_i)
        else:
            A[i, i] = lam[i - 1]*left_basis(u, a_i) + middle_basis(u, a_i) \
                       + (1 - lam[i])*right_basis(u, a_i)

    b = interior_points + fixed
    return np.linalg.solve(A, b)

def get_alpha_array():
    n = len(control_points) - 2
    arr = []
    for i in range(n):
        arr.append(alpha_vars[i].get())
    return arr

def run_kappa_curve_optimization(control_points, iterations):
    m = len(control_points)
    n = m - 2
    if n <= 0:
        return np.zeros((0,2)), np.zeros((0,2)), np.zeros((0,2))

    Q_i_0_set, Q_i_1_set, Q_i_2_set = initialize_Q_points(control_points)
    lam = np.full(n - 1, 0.5) if n > 1 else np.array([])
    t_values = np.full(n, 0.5)

    alpha_array = get_alpha_array()

    for _ in range(iterations):
        if n > 1:
            lam = lambda_i_update(Q_i_0_set, Q_i_1_set, Q_i_2_set, alpha_array)
        update_Q_i_0_Q_i_2(Q_i_0_set, Q_i_1_set, Q_i_2_set, lam)
        t_values = t_i_update_cubic(Q_i_0_set, Q_i_1_set, Q_i_2_set, alpha_array)
        new_Q_i_1 = compute_central_cps_cubic(
            Q_i_0_set, Q_i_1_set, Q_i_2_set,
            lam, t_values,
            interior_points=control_points[1:-1],
            alpha_array=alpha_array
        )
        Q_i_1_set = new_Q_i_1

    return Q_i_0_set, Q_i_1_set, Q_i_2_set

############# Tkinter UI #############

def update_alpha_sliders(Q_i_1_set):
    # Clear existing sliders
    alpha_vars.clear()
    for widget in alpha_frame.winfo_children():
        widget.destroy()
  

    # One alpha for each interior segment
    n = len(control_points) - 2
    for i in range(n):
        var = tk.DoubleVar(value=0.66)
        slider = tk.Scale(alpha_frame, variable=var, from_=0.66, to=0.95,
                          resolution=0.01, orient=tk.HORIZONTAL,
                          label=f"alpha {i+1}", command=on_alpha_change)
        slider.pack(side=tk.LEFT, padx=5)
        alpha_vars.append(var)
    
    draw_alpha_controls(Q_i_1_set)

def draw_curve_fast():
    canvas.delete("curve")

    if show_lines:
        for i in range(len(last_Q_i_0_set)):
            canvas.create_line(last_Q_i_0_set[i][0], last_Q_i_0_set[i][1],
                               last_Q_i_1_set[i][0], last_Q_i_1_set[i][1],
                               fill="gray", tags="curve")
            canvas.create_line(last_Q_i_1_set[i][0], last_Q_i_1_set[i][1],
                               last_Q_i_2_set[i][0], last_Q_i_2_set[i][1],
                               fill="gray", tags="curve")

    alpha_array = get_alpha_array()
    for i in range(len(last_Q_i_0_set)):
        Q0 = last_Q_i_0_set[i]
        Q1 = last_Q_i_1_set[i]
        Q2 = last_Q_i_2_set[i]
        a_i = alpha_array[i]
        cpoints = create_cubic(Q0, Q1, Q2, a_i)

        t_vals = np.linspace(0, 1, 100)
        coords = []
        for t in t_vals:
            p = bezier_cubic(cpoints, t)
            coords.extend(p)
        canvas.create_line(coords, fill="blue", width=2, tags="curve")

    for hit in control_hit_areas:
        canvas.tag_raise(hit)


def draw_curve_full():
    canvas.delete("curve")
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(
        control_points, iterations=iterations_var.get()
    )

    if show_lines:
        for i in range(len(Q_i_0_set)):
            canvas.create_line(Q_i_0_set[i][0], Q_i_0_set[i][1],
                               Q_i_1_set[i][0], Q_i_1_set[i][1],
                               fill="gray", tags="curve")
            canvas.create_line(Q_i_1_set[i][0], Q_i_1_set[i][1],
                               Q_i_2_set[i][0], Q_i_2_set[i][1],
                               fill="gray", tags="curve")

    alpha_array = get_alpha_array()
    for i in range(len(Q_i_0_set)):
        Q0 = Q_i_0_set[i]
        Q1 = Q_i_1_set[i]
        Q2 = Q_i_2_set[i]
        a_i = alpha_array[i]
        cpoints = create_cubic(Q0, Q1, Q2, a_i)

        t_vals = np.linspace(0, 1, 100)
        coords = []
        for t in t_vals:
            p = bezier_cubic(cpoints, t)
            coords.extend(p)
        canvas.create_line(coords, fill="blue", width=2, tags="curve")
        
    for hit in control_hit_areas:
        canvas.tag_raise(hit)
    
    if show_osculating_circles:
        draw_osculating_circles()

    global last_Q_i_0_set, last_Q_i_1_set, last_Q_i_2_set
    last_Q_i_0_set = Q_i_0_set
    last_Q_i_1_set = Q_i_1_set
    last_Q_i_2_set = Q_i_2_set


def draw_q_points(Q_i_0_set, Q_i_1_set, Q_i_2_set):
    canvas.delete("q_points")
    if not q_points_visible:
        return
    for i in range(len(Q_i_1_set)):
        q_pts = [Q_i_0_set[i], Q_i_1_set[i], Q_i_2_set[i]]
        for j, q in enumerate(q_pts):
            q_label = f"Q{i+1},{j}"
            canvas.create_rectangle(q[0] - 5, q[1] - 5, q[0] + 5, q[1] + 5,
                                    outline='black', fill='', tags="q_points")
            canvas.create_text(q[0] + 10, q[1] - 10, text=q_label,
                               fill="black", font=("Arial", 10), tags="q_points")

def toggle_lines():
    global show_lines
    show_lines = not show_lines
    draw_curve_full()

def toggle_label_coords():
    global label_coords_visible
    label_coords_visible = not label_coords_visible
    for coord_label in coord_labels:
        canvas.itemconfig(coord_label, state='normal' if label_coords_visible else 'hidden')

def toggle_q_points():
    global q_points_visible
    q_points_visible = not q_points_visible
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())
    draw_q_points(Q_i_0_set, Q_i_1_set, Q_i_2_set)

def on_drag(event, index):
    control_points[index] = [event.x, event.y]
    canvas.coords(control_circles[index],
                  event.x - CONTROL_RADIUS, event.y - CONTROL_RADIUS,
                  event.x + CONTROL_RADIUS, event.y + CONTROL_RADIUS)
    canvas.coords(control_hit_areas[index],
                  event.x - HIT_RADIUS, event.y - HIT_RADIUS,
                  event.x + HIT_RADIUS, event.y + HIT_RADIUS)
    canvas.coords(control_labels[index], event.x + 10, event.y)
    canvas.coords(coord_labels[index], event.x + 15, event.y + 15)
    canvas.itemconfig(coord_labels[index], text=f"({event.x:.8f}, {event.y:.8f})")
    draw_curve_full()
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())
    draw_q_points(Q_i_0_set, Q_i_1_set, Q_i_2_set)
    if show_osculating_circles:
        draw_full_osculating_ui()


    

def on_alpha_change(val):
    draw_curve_full()



def add_control_point():
    global control_points, control_circles, control_hit_areas, control_labels, coord_labels

    # 1) Create and append the new point
    new_point = np.array([canvas_width / 2, canvas_height / 2])
    control_points = np.vstack((control_points, new_point))

    # 2) Create the associated canvas items
    hit_area = canvas.create_oval(new_point[0] - HIT_RADIUS, new_point[1] - HIT_RADIUS,
                                  new_point[0] + HIT_RADIUS, new_point[1] + HIT_RADIUS,
                                  fill="", outline="")
    control_hit_areas.append(hit_area)

    circle = canvas.create_oval(new_point[0] - CONTROL_RADIUS, new_point[1] - CONTROL_RADIUS,
                                new_point[0] + CONTROL_RADIUS, new_point[1] + CONTROL_RADIUS,
                                fill='red')
    control_circles.append(circle)

    new_index = len(control_points)
    label = canvas.create_text(new_point[0] + 10, new_point[1],
                               text=str(new_index), fill="black", font=("Arial", 12))
    control_labels.append(label)

    coord_label = canvas.create_text(new_point[0] + 15, new_point[1] + 15,
                                     text=f"({new_point[0]:.8f}, {new_point[1]:.8f})",
                                     fill="blue", font=("Arial", 10),
                                     state='normal' if label_coords_visible else 'hidden')
    coord_labels.append(coord_label)

    # Bind dragging events
    canvas.tag_bind(hit_area, "<B1-Motion>", lambda event, idx=new_index-1: on_drag(event, idx))
    canvas.tag_bind(circle, "<B1-Motion>", lambda event, idx=new_index-1: on_drag(event, idx))

    update_alpha_sliders(np.zeros((len(control_points) - 2, 2)))  # pass dummy Q_i_1 shape
    Q_i_0, Q_i_1, Q_i_2 = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())
    draw_q_points(Q_i_0, Q_i_1, Q_i_2)
    draw_curve_full()
    if show_osculating_circles:
        draw_full_osculating_ui()



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
        draw_curve_full()
        Q_i_0, Q_i_1, Q_i_2 = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())
        update_alpha_sliders(Q_i_1)
        draw_q_points(Q_i_0, Q_i_1, Q_i_2)
        if show_osculating_circles:
            draw_full_osculating_ui()


        message_label.config(text="")
    else:
        message_label.config(text="Minimum of 5 control points required.")
        window.after(5000, lambda: message_label.config(text=""))

def draw_alpha_controls(Q_i_1_set):
    
    for item in alpha_circle_items + alpha_dot_items:
        canvas.delete(item)
    alpha_circle_items.clear()
    alpha_dot_items.clear()

    for i, center in enumerate(Q_i_1_set):
        cx, cy = center

       
        alpha_val = alpha_vars[i].get()
        alpha_val = max(0.66, min(alpha_val, 0.95))
        alpha_vars[i].set(alpha_val)

        
        fraction = (alpha_val - 0.66) / (0.95 - 0.66)
        angle = 2 * math.pi * fraction

        
        dot_x = cx + alpha_circle_radius * math.cos(angle)
        dot_y = cy + alpha_circle_radius * math.sin(angle)

        
        circle_item = canvas.create_oval(cx - alpha_circle_radius,
                                         cy - alpha_circle_radius,
                                         cx + alpha_circle_radius,
                                         cy + alpha_circle_radius,
                                         outline="green", dash=(3, 3))
        alpha_circle_items.append(circle_item)

        # Draw the dot
        dot_item = canvas.create_oval(dot_x - 5, dot_y - 5,
                                      dot_x + 5, dot_y + 5,
                                      fill="green", outline="")
        alpha_dot_items.append(dot_item)

        # Bind a motion callback so dragging this dot updates alpha
        canvas.tag_bind(dot_item, "<B1-Motion>", lambda e, idx=i: on_drag_alpha_dot(e, idx))

def on_drag_alpha_dot(event, i):
    center = last_Q_i_1_set[i]
 # ⚡️ NO recomputation. Use the last Q-set.

    cx, cy = center

    # Vector from center to mouse
    dx = event.x - cx
    dy = event.y - cy

    # Angle in [-π, π]
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += 2 * math.pi

    fraction = angle / (2 * math.pi)
    alpha_val = 0.66 + fraction * (0.95 - 0.66)
    alpha_vars[i].set(alpha_val)

    # Update the dot position visually
    dot_x = cx + alpha_circle_radius * math.cos(angle)
    dot_y = cy + alpha_circle_radius * math.sin(angle)
    dot_item = alpha_dot_items[i]
    canvas.coords(dot_item, dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5)

    # Now redraw the curve
    draw_curve_full()
    if show_osculating_circles:
        draw_only_osculating_circles()

def draw_full_osculating_ui():
    draw_only_osculating_circles()
    create_osculating_sliders()



######### Osculating Circle ################

# Function to calculate the first derivative of the Bézier curve at parameter t
def bezier_first_derivative(control_points, t):
    n = len(control_points) - 1
    derivative = np.zeros(2)
    for i in range(n):
        derivative += n * (control_points[i + 1] - control_points[i]) \
                      * ((1 - t)**(n - 1 - i)) * (t**i)
    return derivative

# Function to calculate the second derivative of the Bézier curve at parameter t
def bezier_second_derivative(control_points, t):
    n = len(control_points) - 1
    second_derivative = np.zeros(2)
    one_minus_t = 1 - t
    t_power = 1.0
    for i in range(n - 1):
        second_derivative += n*(n - 1) * (control_points[i+2] - 2*control_points[i+1] + control_points[i]) \
                             * (one_minus_t**(n - 2 - i)) * t_power
        t_power *= t
    return second_derivative


# Function to calculate curvature and its sign at parameter t
def curvature_and_sign(control_points, t):
    d1 = bezier_first_derivative(control_points, t)
    d2 = bezier_second_derivative(control_points, t)
    d1_norm = np.linalg.norm(d1)
    cross_product = d1[0] * d2[1] - d1[1] * d2[0]  # This determines the sign of the curvature
    curvature = abs(cross_product) / (d1_norm ** 3) if d1_norm != 0 else 0
    return curvature, np.sign(cross_product)

# Storage for drawn circles/lines/labels
osculating_circles = []
osculating_lines = []
osculating_labels = []
show_osculating_circles = False
osculating_data = []  
osculating_dots = [] 

def draw_only_osculating_circles():
    global osculating_circles

    # Clear only the osculating circles
    for item in osculating_circles:
        canvas.delete(item)
    osculating_circles.clear()

    # Recompute updated circles
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(
        control_points, iterations=iterations_var.get()
    )
    alpha_array = get_alpha_array()
    t_values = t_i_update_cubic(Q_i_0_set, Q_i_1_set, Q_i_2_set, alpha_array)

    for i in range(len(Q_i_0_set)):
        cpoints = create_cubic(Q_i_0_set[i], Q_i_1_set[i], Q_i_2_set[i], alpha_array[i])
        t_val = t_values[i]
        pt = bezier_cubic(cpoints, t_val)

        d1 = bezier_first_derivative(cpoints, t_val)
        norm_d1 = np.linalg.norm(d1)
        if norm_d1 == 0:
            continue
        d1 /= norm_d1

        kappa, sign_cp = curvature_and_sign(cpoints, t_val)
        if kappa == 0:
            continue
        radius = 1.0 / kappa

        normal_vec = np.array([-d1[1], d1[0]]) * sign_cp
        center = pt + radius * normal_vec

        circle_item = canvas.create_oval(
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius,
            outline="orange"
        )
        osculating_circles.append(circle_item)


def draw_osculating_circles():

    global osculating_circles, osculating_lines, osculating_labels
    global osculating_data, osculating_dots

    if osculating_dots:
        return

    # Get the optimized cubic segments
    Q_i_0_set, Q_i_1_set, Q_i_2_set = run_kappa_curve_optimization(
        control_points, iterations=iterations_var.get()
    )
    alpha_array = get_alpha_array()
    t_values = t_i_update_cubic(Q_i_0_set, Q_i_1_set, Q_i_2_set, alpha_array)

    line_half_length = 50  # Half-length of normal vector line for alpha slider

    for i in range(len(Q_i_0_set)):
        # Build cubic control points, t_val, pt
        cpoints = create_cubic(Q_i_0_set[i], Q_i_1_set[i], Q_i_2_set[i], alpha_array[i])
        t_val = t_values[i]
        pt = bezier_cubic(cpoints, t_val)

        # First derivative and normal
        d1 = bezier_first_derivative(cpoints, t_val)
        norm_d1 = np.linalg.norm(d1)
        if norm_d1 == 0:
            continue
        d1 /= norm_d1
        

        # Compute curvature
        kappa, sign_cp = curvature_and_sign(cpoints, t_val)
        if kappa == 0:
            continue
        radius = 1.0 / kappa

        normal_vec = np.array([-d1[1], d1[0]]) * sign_cp  # unit normal vector

        # Compute circle center
        center = pt + radius * normal_vec

        # --- 1. Draw Osculating Circle ---
        circle_item = canvas.create_oval(
            center[0] - radius, center[1] - radius,
            center[0] + radius, center[1] + radius,
            outline="orange"
        )
        osculating_circles.append(circle_item)

def create_osculating_sliders():
    # Only create if not already present
    if osculating_dots:
        return

    Q_i_0_set, Q_i_1_set, Q_i_2_set = last_Q_i_0_set, last_Q_i_1_set, last_Q_i_2_set
    alpha_array = get_alpha_array()
    t_values = t_i_update_cubic(Q_i_0_set, Q_i_1_set, Q_i_2_set, alpha_array)

    line_half_length = 50

    for i in range(len(Q_i_0_set)):
        cpoints = create_cubic(Q_i_0_set[i], Q_i_1_set[i], Q_i_2_set[i], alpha_array[i])
        t_val = t_values[i]
        pt = bezier_cubic(cpoints, t_val)

        d1 = bezier_first_derivative(cpoints, t_val)
        norm_d1 = np.linalg.norm(d1)
        if norm_d1 == 0:
            continue
        d1 /= norm_d1
        normal_vec = np.array([-d1[1], d1[0]])
        normal_vec *= np.sign(curvature_and_sign(cpoints, t_val)[1])

        # Slider geometry
        p_start = pt - line_half_length * normal_vec
        p_end = pt + line_half_length * normal_vec
        osculating_data.append((p_start, p_end))

        # Line
        line_item = canvas.create_line(p_start[0], p_start[1], p_end[0], p_end[1], fill="orange", width=2)
        osculating_lines.append(line_item)

        # Dot
        fraction = (alpha_array[i] - 0.66) / (0.95 - 0.66)
        dot_x = p_start[0] + fraction * (p_end[0] - p_start[0])
        dot_y = p_start[1] + fraction * (p_end[1] - p_start[1])
        dot_item = canvas.create_oval(dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5, fill="orange", outline="")
        canvas.tag_bind(dot_item, "<B1-Motion>", lambda e, idx=i: on_drag_osculating_dot(e, idx))
        osculating_dots.append(dot_item)

def on_drag_osculating_dot(event, i):
    if i >= len(osculating_data):
        return

    p_start, p_end = osculating_data[i]
    dot_item = osculating_dots[i]

    vec_x = p_end[0] - p_start[0]
    vec_y = p_end[1] - p_start[1]
    vec_len_sq = vec_x*vec_x + vec_y*vec_y
    if vec_len_sq < 1e-12:
        return

    mvec_x = event.x - p_start[0]
    mvec_y = event.y - p_start[1]

    dot_val = mvec_x*vec_x + mvec_y*vec_y
    fraction = dot_val / vec_len_sq
    fraction = max(0.0, min(1.0, fraction))

    alpha_val = 0.66 + fraction * (0.95 - 0.66)
    alpha_vars[i].set(alpha_val)

    dot_x = p_start[0] + fraction * vec_x
    dot_y = p_start[1] + fraction * vec_y
    canvas.coords(dot_item, dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5)

    draw_curve_full()  # <<< Use fast redraw only



def toggle_osculating_circles():
    global show_osculating_circles
    show_osculating_circles = not show_osculating_circles

    if show_osculating_circles:
        draw_only_osculating_circles()
        create_osculating_sliders()
    else:
        for item in osculating_circles + osculating_lines + osculating_dots:
            canvas.delete(item)
        osculating_circles.clear()
        osculating_lines.clear()
        osculating_dots.clear()
        osculating_data.clear()


######## Main Tkinter setup #########

window = tk.Tk()
window.title("Cubic κ-Curve with Per-Segment Alpha (0.66–0.95)")

canvas_width = 800
canvas_height = 600
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
canvas.pack()

control_circles = []
control_labels = []
coord_labels = []

iterations_var = tk.IntVar(value=10)
iterations_spinbox = tk.Spinbox(window, from_=1, to=1000, increment=1,
                                textvariable=iterations_var, command=draw_curve_full, width=5)
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

toggle_circle_button = tk.Button(window, text="Toggle Osculating Circles", command=toggle_osculating_circles)
toggle_circle_button.pack(pady=5)

alpha_frame = tk.Frame(window)
alpha_frame.pack(pady=5)

message_label = tk.Label(window, text="", fg="red")
message_label.pack(pady=5)





# Initial control points
control_points = np.array([
    [50, 400],
    [250, 200],
    [350, 300],
    [450, 400],
    [550, 350]
], dtype=float)


alpha_vars = []
n = len(control_points) - 2  # number of interior segments
for i in range(n):
    var = tk.DoubleVar(value=0.66)
    alpha_vars.append(var)

alpha_circle_items = []  
alpha_dot_items = []     
alpha_circle_radius = 50 

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
    canvas.tag_bind(hit_area, "<B1-Motion>", lambda e, idx=i: on_drag(e, idx))
    canvas.tag_bind(circle, "<B1-Motion>", lambda e, idx=i: on_drag(e, idx))
    coord_label = canvas.create_text(point[0] + 15, point[1] + 15,
                                     text=f"({point[0]:.8f}, {point[1]:.8f})",
                                     fill="blue", font=("Arial", 10),
                                     state='hidden')
    coord_labels.append(coord_label)



draw_curve_full()
Q_i_0, Q_i_1, Q_i_2 = run_kappa_curve_optimization(control_points, iterations=iterations_var.get())


window.mainloop()
