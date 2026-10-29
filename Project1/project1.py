from pathlib import Path
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
import pyCAPS

# ============================================================
# SETUP / FILE PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
CSM_FILE = PROJECT_DIR / 'aircraft.csm'
CAPS_RUN_DIR = PROJECT_DIR / 'caps_run'
FIGURE_DIR = PROJECT_DIR / 'figures'
FIGURE_DIR.mkdir(exist_ok=True)
os.chdir(PROJECT_DIR)
if CAPS_RUN_DIR.exists():
    shutil.rmtree(CAPS_RUN_DIR)
caps_problem = pyCAPS.Problem(problemName=str(CAPS_RUN_DIR), capsFile=str(CSM_FILE), outLevel=0)
plane = caps_problem.geometry

import subprocess

XFOIL_PATH = "/Users/jadenherman/software/ESP129/EngSketchPad/bin/xfoil"
POLAR_DIR = PROJECT_DIR / "xfoil_polars"
POLAR_DIR.mkdir(exist_ok=True)

RE_VALUES = [100000, 200000, 300000, 400000, 500000, 600000, 800000, 1000000]

# ============================================================
# CONSTANTS
# ============================================================

g = 9.80665
rho_sl = 1.225
mu_sl = 1.789e-05
rho_foam = 30.0
motor_mass = 0.08
prop_mass = 0.019
esc_mass = 0.082
battery_mass = 0.348
receiver_mass = 0.0145
servo_mass = 0.0166
oswald_efficiency = 0.8
cd_wing_profile = 0.01
cd_tail_profile = 0.015

# ============================================================
# HELPERS
# ============================================================

def despmtr(name):
    return plane.despmtr[name].value

def outpmtr(name):
    return plane.outpmtr[name].value

# ============================================================
# WEIGHTS MODEL
# ============================================================

def weights_model():
    wing_x = despmtr('wing_x')
    wing_chord = outpmtr('wing_chord')
    nose_volume = outpmtr('noseV')
    nose_xcg = outpmtr('nosecg')
    fuse_volume = outpmtr('fuseV')
    fuse_xcg = outpmtr('fusecg')
    wing_volume = outpmtr('wingV')
    wing_xcg = outpmtr('wingcg')
    htail_volume = outpmtr('htailV')
    htail_xcg = outpmtr('htailcg')
    vtail_volume = outpmtr('vtailV')
    vtail_xcg = outpmtr('vtailcg')
    motor_xcg = outpmtr('motorcg')
    prop_xcg = outpmtr('propcg')
    esc_xcg = outpmtr('esccg')
    battery_xcg = outpmtr('battcg')
    receiver_xcg = outpmtr('rxcg')
    ail_left_xcg = outpmtr('ail_left_cg')
    ail_right_xcg = outpmtr('ail_right_cg')
    elevator_xcg = outpmtr('elev_cg')
    rudder_xcg = outpmtr('rud_cg')
    esc_volume = outpmtr('escV')
    battery_volume = outpmtr('battV')
    receiver_volume = outpmtr('rxV')
    servo_volume = outpmtr('servoV')
    parts = []

    def add(name, mass, x):
        parts.append((name, mass, x))
    add('Nose foam', rho_foam * nose_volume, nose_xcg)
    add('Fuselage foam', rho_foam * fuse_volume, fuse_xcg)
    add('Main wing foam', rho_foam * wing_volume, wing_xcg)
    add('Horizontal tail foam', rho_foam * htail_volume, htail_xcg)
    add('Vertical tail foam', rho_foam * vtail_volume, vtail_xcg)
    add('Motor', motor_mass, motor_xcg)
    add('Propeller', prop_mass, prop_xcg)
    add('ESC', esc_mass, esc_xcg)
    add('Battery', battery_mass, battery_xcg)
    add('Receiver', receiver_mass, receiver_xcg)
    add('Left aileron servo', servo_mass, ail_left_xcg)
    add('Right aileron servo', servo_mass, ail_right_xcg)
    add('Elevator servo', servo_mass, elevator_xcg)
    add('Rudder servo', servo_mass, rudder_xcg)
    add('ESC foam removed', -rho_foam * esc_volume, esc_xcg)
    add('Battery foam removed', -rho_foam * battery_volume, battery_xcg)
    add('Receiver foam removed', -rho_foam * receiver_volume, receiver_xcg)
    add('Left servo foam removed', -rho_foam * servo_volume, ail_left_xcg)
    add('Right servo foam removed', -rho_foam * servo_volume, ail_right_xcg)
    add('Elevator foam removed', -rho_foam * servo_volume, elevator_xcg)
    add('Rudder foam removed', -rho_foam * servo_volume, rudder_xcg)
    total_mass = sum((p[1] for p in parts))
    total_moment = sum((p[1] * p[2] for p in parts))
    xcg = total_moment / total_mass
    weight = total_mass * g
    cg_fraction = (xcg - wing_x) / wing_chord
    return {'mass': total_mass, 'weight': weight, 'xcg': xcg, 'cg_fraction': cg_fraction, 'cg_percent': 100 * cg_fraction, 'parts': parts}

# ============================================================
# AERODYNAMIC MODEL
# ============================================================

def generate_xfoil_polars():
    for Re in RE_VALUES:
        filename = f"naca2412_Re{Re}.txt"
        polar_file = POLAR_DIR / filename

        if polar_file.exists():
            continue

        commands = f"""NACA 2412
PANE
OPER
VISC {Re}
ITER 200
PACC
{filename}

ASEQ -4 16 0.5
PACC

QUIT
"""

        result = subprocess.run(
            [XFOIL_PATH],
            input=commands,
            text=True,
            cwd=POLAR_DIR,
            capture_output=True
        )

        if not polar_file.exists():
            print(f"XFOIL FAILED FOR Re = {Re}")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"XFOIL did not create {filename}")

        print(f"Generated {filename}")


def load_xfoil_polar(file):
    rows = []

    with open(file, "r") as f:
        for line in f:
            values = line.split()

            if len(values) < 3:
                continue

            try:
                alpha = float(values[0])
                CL = float(values[1])
                CD = float(values[2])
                rows.append((alpha, CL, CD))
            except ValueError:
                continue

    data = np.array(rows)

    return data[:, 0], data[:, 1], data[:, 2]


def load_all_polars():
    polars = {}

    for Re in RE_VALUES:
        file = POLAR_DIR / f"naca2412_Re{Re}.txt"
        alpha, CL, CD = load_xfoil_polar(file)

        # sort by CL for interpolation
        order = np.argsort(CL)

        polars[Re] = {
            "alpha": alpha[order],
            "CL": CL[order],
            "CD": CD[order]
        }

    return polars

generate_xfoil_polars()
naca2412_polars = load_all_polars()

def naca2412_cd(CL_target, Re_target):
    CL_target = np.asarray(CL_target)
    Re_target = np.asarray(Re_target)

    CL_target, Re_target = np.broadcast_arrays(CL_target, Re_target)

    CD = np.zeros_like(CL_target, dtype=float)

    for i in np.ndindex(CL_target.shape):
        cl = CL_target[i]
        re = Re_target[i]

        if re <= RE_VALUES[0]:
            Re1 = Re2 = RE_VALUES[0]
        elif re >= RE_VALUES[-1]:
            Re1 = Re2 = RE_VALUES[-1]
        else:
            Re1 = max(r for r in RE_VALUES if r <= re)
            Re2 = min(r for r in RE_VALUES if r >= re)

        p1 = naca2412_polars[Re1]
        cd1 = np.interp(cl, p1["CL"], p1["CD"])

        if Re1 == Re2:
            CD[i] = cd1
            continue

        p2 = naca2412_polars[Re2]
        cd2 = np.interp(cl, p2["CL"], p2["CD"])

        frac = (re - Re1) / (Re2 - Re1)
        CD[i] = cd1 + frac*(cd2 - cd1)

    return CD

def aero_model(V, weight, rho=rho_sl):
    V = np.asarray(V)
    S = despmtr('wing_area')
    AR = despmtr('wing_AR')
    fus_length = despmtr('fus_length')
    fus_diameter = outpmtr('fus_diameter')
    ht_area = outpmtr('ht_area')
    vt_area = outpmtr('vt_area')
    q = 0.5 * rho * V ** 2
    CL = weight / (q * S)
    lift = q * S * CL
    wing_chord = outpmtr("wing_chord")
    mu = mu_sl
    Re = rho * V * wing_chord / mu
    CD_wing = naca2412_cd(CL, Re)
    fineness_ratio = fus_length / fus_diameter
    CD_fuse = fus_length * fus_diameter / S * (0.25 * fineness_ratio ** (-0.9) + 0.05 * fineness_ratio ** 0.57)
    CD_fuse = CD_fuse * np.ones_like(V)
    CD_ht = cd_tail_profile * ht_area / S
    CD_vt = cd_tail_profile * vt_area / S
    CD_ht = CD_ht * np.ones_like(V)
    CD_vt = CD_vt * np.ones_like(V)
    CD_induced = CL ** 2 / (np.pi * oswald_efficiency * AR)
    CD_total = CD_wing + CD_fuse + CD_ht + CD_vt + CD_induced
    drag = q * S * CD_total
    return {"Re": Re, 'V': V, 'q': q, 'CL': CL, 'lift': lift, 'CD_wing': CD_wing, 'CD_fuse': CD_fuse, 'CD_ht': CD_ht, 'CD_vt': CD_vt, 'CD_induced': CD_induced, 'CD_total': CD_total, 'drag': drag}

# ============================================================
# PROPULSION MODEL
# ============================================================

def propulsion_model(V, throttle, rho=rho_sl):
    V = np.asarray(V)
    thrust = 15.0 * (rho / rho_sl) * (1.0 - V / 50.0) * throttle
    thrust = np.maximum(thrust, 0.0)
    efficiency = 0.7
    return {'thrust': thrust, 'efficiency': efficiency}

# ============================================================
# BASELINE AIRCRAFT
# ============================================================

weights = weights_model()
print()
print('=' * 72)
print('BASELINE AIRCRAFT')
print('=' * 72)
print(f"Mass:             {weights['mass']:.4f} kg")
print(f"Weight:           {weights['weight']:.4f} N")
print(f"CG from nose:     {weights['xcg']:.4f} m")
print(f"CG location:      {weights['cg_percent']:.2f}% chord")

# ============================================================
# STEP 5 - AERODYNAMIC CHECK
# ============================================================

V_test = 20.0
aero_test = aero_model(V_test, weights['weight'])
print()
print('=' * 72)
print('STEP 5 - AERODYNAMIC MODEL CHECK AT 20 m/s')
print('=' * 72)
print(f"CL:               {float(aero_test['CL']):.4f}")
print(f"CD wing profile:  {float(aero_test['CD_wing']):.4f}")
print(f"CD fuselage:      {float(aero_test['CD_fuse']):.4f}")
print(f"CD horizontal:    {float(aero_test['CD_ht']):.4f}")
print(f"CD vertical:      {float(aero_test['CD_vt']):.4f}")
print(f"CD induced:       {float(aero_test['CD_induced']):.4f}")
print(f"CD total:         {float(aero_test['CD_total']):.4f}")
print(f"Drag:             {float(aero_test['drag']):.4f} N")

# ============================================================
# STEP 6 - DRAG / THRUST PLOT
# ============================================================

V = np.linspace(4, 49.0, 400)
aero = aero_model(V, weights['weight'])
T20 = propulsion_model(V, 0.2)['thrust']
T50 = propulsion_model(V, 0.5)['thrust']
T100 = propulsion_model(V, 1.0)['thrust']

plt.figure(figsize=(8, 5))
plt.plot(V, aero['drag'], label='Drag Required')
plt.plot(V, T20, label='20% Throttle')
plt.plot(V, T50, label='50% Throttle')
plt.plot(V, T100, label='100% Throttle')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Force (N)')
plt.xlim(4, 49)
plt.title('Drag Required and Thrust Available at Sea Level')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'step6_drag_thrust.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# STEP 7 - PERFORMANCE
# ============================================================

CL_max = 1.5
battery_voltage = 14.8
battery_capacity_Ah = 3.3
battery_energy_Wh = battery_voltage * battery_capacity_Ah
battery_energy_J = battery_energy_Wh * 3600.0
powertrain_efficiency = 0.7

def atmosphere(h):
    T0 = 288.15
    p0 = 101325.0
    rho0 = 1.225
    R = 287.05
    g0 = 9.80665
    if h <= 11000.0:
        L = 0.0065
        T = T0 - L * h
        p = p0 * (T / T0) ** (g0 / (R * L))
    else:
        T11 = 216.65
        p11 = p0 * (T11 / T0) ** (g0 / (R * 0.0065))
        T = T11
        p = p11 * np.exp(-g0 * (h - 11000.0) / (R * T))
    rho = p / (R * T)
    return rho

W = weights['weight']
S = despmtr('wing_area')
V_stall_SL = np.sqrt(2.0 * W / (rho_sl * S * CL_max))
V_perf = np.linspace(V_stall_SL, 49.0, 3000)
aero_perf = aero_model(V_perf, W, rho_sl)
D_perf = aero_perf['drag']
P_flight = D_perf * V_perf
P_battery = P_flight / powertrain_efficiency
i_range = np.argmin(D_perf)
V_best_range = V_perf[i_range]
D_best_range = D_perf[i_range]
best_range = powertrain_efficiency * battery_energy_J / D_best_range
i_endurance = np.argmin(P_flight)
V_best_endurance = V_perf[i_endurance]
P_min_flight = P_flight[i_endurance]
P_min_battery = P_battery[i_endurance]
best_endurance_sec = battery_energy_J / P_min_battery
best_endurance_min = best_endurance_sec / 60.0
altitudes = np.linspace(0.0, 20000.0, 1001)
max_ROC = np.zeros_like(altitudes)
best_climb_speed = np.zeros_like(altitudes)

for i, h in enumerate(altitudes):
    rho = atmosphere(h)
    V_stall_h = np.sqrt(2.0 * W / (rho * S * CL_max))
    V_h = np.linspace(V_stall_h, 49.0, 800)
    aero_h = aero_model(V_h, W, rho)
    thrust_h = propulsion_model(V_h, 1.0, rho)['thrust']
    ROC = (thrust_h - aero_h['drag']) * V_h / W
    i_max = np.argmax(ROC)
    max_ROC[i] = ROC[i_max]
    best_climb_speed[i] = V_h[i_max]
valid_absolute = np.where(max_ROC >= 0.0)[0]

if len(valid_absolute) > 0:
    absolute_ceiling = altitudes[valid_absolute[-1]]
else:
    absolute_ceiling = np.nan
service_ROC = 0.508
valid_service = np.where(max_ROC >= service_ROC)[0]
if len(valid_service) > 0:
    service_ceiling = altitudes[valid_service[-1]]
else:
    service_ceiling = np.nan
    
bank_deg = np.linspace(0.0, 80.0, 161)
bank_rad = np.deg2rad(bank_deg)
load_factor = 1.0 / np.cos(bank_rad)
V_stall_bank = np.sqrt(2.0 * load_factor * W / (rho_sl * S * CL_max))
V_dash_bank = np.full_like(bank_deg, np.nan)

for i, n in enumerate(load_factor):
    V_min = V_stall_bank[i]
    V_test = np.linspace(V_min, 49.0, 2000)
    aero_turn = aero_model(V_test, n * W, rho_sl)
    T_turn = propulsion_model(V_test, 1.0, rho_sl)['thrust']
    feasible = np.where(T_turn >= aero_turn['drag'])[0]
    if len(feasible) > 0:
        V_dash_bank[i] = V_test[feasible[-1]]
turn_radius = np.full_like(bank_deg, np.nan)

for i, phi in enumerate(bank_rad):
    if phi == 0.0:
        continue
    V_turn = V_stall_bank[i]
    n = load_factor[i]
    aero_turn = aero_model(V_turn, n * W, rho_sl)
    thrust_turn = propulsion_model(V_turn, 1.0, rho_sl)['thrust']
    if thrust_turn >= aero_turn['drag']:
        turn_radius[i] = V_turn ** 2 / (g * np.tan(phi))

if np.any(np.isfinite(turn_radius)):
    i_min_radius = np.nanargmin(turn_radius)
    minimum_turn_radius = turn_radius[i_min_radius]
    min_radius_bank = bank_deg[i_min_radius]
    min_radius_speed = V_stall_bank[i_min_radius]
else:
    minimum_turn_radius = np.nan
    min_radius_bank = np.nan
    min_radius_speed = np.nan

print()
print('=' * 72)
print('STEP 7 - PERFORMANCE')
print('=' * 72)
print()
print('BEST RANGE')
print('-' * 72)
print(f'Best range speed:       {V_best_range:.3f} m/s')
print(f'Minimum drag:           {D_best_range:.3f} N')
print(f'Best range:             {best_range / 1000:.3f} km')
print()
print('BEST ENDURANCE')
print('-' * 72)
print(f'Best endurance speed:   {V_best_endurance:.3f} m/s')
print(f'Minimum flight power:   {P_min_flight:.3f} W')
print(f'Battery power required: {P_min_battery:.3f} W')
print(f'Best endurance:         {best_endurance_min:.2f} min')
print()
print('CEILINGS')
print('-' * 72)
print(f'Service ceiling:        {service_ceiling:.0f} m')
print(f'Absolute ceiling:       {absolute_ceiling:.0f} m')
print()
print('STALL')
print('-' * 72)
print(f'Sea-level stall speed:  {V_stall_SL:.3f} m/s')
print()
print('MINIMUM TURN RADIUS')
print('-' * 72)
print(f'Minimum turn radius:    {minimum_turn_radius:.3f} m')
print(f'Bank angle:             {min_radius_bank:.1f} deg')
print(f'Turn speed:             {min_radius_speed:.3f} m/s')
plt.figure(figsize=(7, 5))
plt.plot(bank_deg, V_stall_bank)
plt.xlabel('Bank Angle (deg)')
plt.ylabel('Stall Speed (m/s)')
plt.title('Stall Speed vs Bank Angle')
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'step7_stall_speed_vs_bank.png', dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(7, 5))
plt.plot(bank_deg, V_dash_bank)
plt.xlabel('Bank Angle (deg)')
plt.ylabel('Dash Speed (m/s)')
plt.title('Dash Speed vs Bank Angle')
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'step7_dash_speed_vs_bank.png', dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(7, 5))
plt.plot(altitudes, max_ROC)
plt.axhline(0.508, linestyle='--', label='Service Ceiling Criterion')
plt.axhline(0.0, linestyle='--', label='Absolute Ceiling Criterion')
plt.xlabel('Altitude (m)')
plt.ylabel('Maximum Rate of Climb (m/s)')
plt.title('Maximum Rate of Climb vs Altitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'step7_ceiling.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# STEP 8 - WING STRUCTURAL LIMIT
# ============================================================

sigma_tensile = 140000.0
sigma_compressive = 100000.0
sigma_allow = min(sigma_tensile, sigma_compressive)
I_norm_2412 = 6.84e-05
b = outpmtr('wing_span')
c_wing = outpmtr('wing_chord')
I_wing = I_norm_2412 * c_wing ** 4
c_bending = 0.06 * c_wing
y_lift = 4.0 * b / (6.0 * np.pi)
root_moment_1g = W / 2.0 * y_lift
stress_1g = root_moment_1g * c_bending / I_wing
n_structural_max = sigma_allow / stress_1g

if n_structural_max >= 1.0:
    phi_structural_max = np.degrees(np.arccos(1.0 / n_structural_max))
    structural_feasible = True
else:
    phi_structural_max = np.nan
    structural_feasible = False

print()
print('=' * 72)
print('STEP 8 - WING STRUCTURAL LIMIT')
print('=' * 72)
print(f'Wing span:                   {b:.4f} m')
print(f'Wing chord:                  {c_wing:.4f} m')
print(f'Bending moment of inertia:   {I_wing:.6e} m^4')
print(f'Half-wing lift arm:          {y_lift:.4f} m')
print(f'1-g root bending moment:     {root_moment_1g:.4f} N*m')
print(f'1-g bending stress:          {stress_1g / 1000:.2f} kPa')
print(f'Allowable bending stress:    {sigma_allow / 1000:.2f} kPa')
print(f'Maximum structural load n:   {n_structural_max:.3f}')

if structural_feasible:
    print(f'Maximum structural bank:     {phi_structural_max:.2f} deg')
else:
    print('Maximum structural bank:     N/A')
    print('STRUCTURAL CHECK:             FAILS BELOW 1 g')
    print('The baseline wing cannot support steady level flight under the stated foam properties and beam model.')

# ============================================================
# STEP 9 - DESIGN TRADE STUDY
# ============================================================

wing_areas = [0.25, 0.6, 1.0]
aspect_ratios = [2.0, 5.0, 8.0]
trade_results = []

def structural_model(weight):
    b = outpmtr('wing_span')
    c = outpmtr('wing_chord')
    I = 6.84e-05 * c ** 4
    c_bending = 0.06 * c
    y_lift = 4.0 * b / (6.0 * np.pi)
    M_1g = weight / 2.0 * y_lift
    stress_1g = M_1g * c_bending / I
    sigma_allow = 100000.0
    n_max = sigma_allow / stress_1g
    return n_max

def endurance_model(weight):
    S = despmtr('wing_area')
    V_stall = np.sqrt(2.0 * weight / (rho_sl * S * CL_max))
    V_test = np.linspace(V_stall, 49.0, 2000)
    aero = aero_model(V_test, weight, rho_sl)
    power_flight = aero['drag'] * V_test
    i_best = np.argmin(power_flight)
    V_best = V_test[i_best]
    P_best = power_flight[i_best]
    P_battery = P_best / powertrain_efficiency
    endurance_sec = battery_energy_J / P_battery
    return (V_best, endurance_sec / 60.0)

def level_flight_feasible(weight):
    S = despmtr('wing_area')
    V_stall = np.sqrt(2.0 * weight / (rho_sl * S * CL_max))
    V_test = np.linspace(V_stall, 49.0, 1500)
    aero = aero_model(V_test, weight, rho_sl)
    thrust = propulsion_model(V_test, 1.0, rho_sl)['thrust']
    return np.any(thrust >= aero['drag'])

print()
print('=' * 90)
print('STEP 9 - WING AREA / ASPECT RATIO TRADE STUDY')
print('=' * 90)

for S_trade in wing_areas:
    for AR_trade in aspect_ratios:
        plane.despmtr['wing_area'].value = S_trade
        plane.despmtr['wing_AR'].value = AR_trade
        trade_weights = weights_model()
        trade_mass = trade_weights['mass']
        trade_weight = trade_weights['weight']
        trade_cg = trade_weights['cg_percent']
        V_endurance, endurance_min = endurance_model(trade_weight)
        n_max_trade = structural_model(trade_weight)
        cg_ok = 25.0 <= trade_cg <= 50.0
        structure_ok = n_max_trade >= 1.0
        flight_ok = level_flight_feasible(trade_weight)
        feasible = cg_ok and structure_ok and flight_ok
        trade_results.append({'S': S_trade, 'AR': AR_trade, 'mass': trade_mass, 'cg': trade_cg, 'V_end': V_endurance, 'endurance': endurance_min, 'n_max': n_max_trade, 'cg_ok': cg_ok, 'structure_ok': structure_ok, 'flight_ok': flight_ok, 'feasible': feasible})
print()
print(f"{'S':>6}{'AR':>6}{'Mass':>10}{'CG %c':>10}{'V_end':>10}{'End min':>12}{'n_max':>10}{'CG':>8}{'Struct':>9}{'Flight':>9}{'Feasible':>11}")
print('-' * 101)
for r in trade_results:
    print(f"{r['S']:6.2f}{r['AR']:6.1f}{r['mass']:10.3f}{r['cg']:10.2f}{r['V_end']:10.2f}{r['endurance']:12.1f}{r['n_max']:10.2f}{str(r['cg_ok']):>8}{str(r['structure_ok']):>9}{str(r['flight_ok']):>9}{str(r['feasible']):>11}")
best_trade = max(trade_results, key=lambda r: r['endurance'])
print()
print('BEST THEORETICAL ENDURANCE DESIGN')
print('-' * 72)
print(f"Wing area:           {best_trade['S']:.2f} m^2")
print(f"Aspect ratio:        {best_trade['AR']:.1f}")
print(f"Endurance:           {best_trade['endurance']:.1f} min")
print(f"Best endurance V:    {best_trade['V_end']:.2f} m/s")
print(f"Maximum load factor: {best_trade['n_max']:.2f}")
print(f"Feasible:            {best_trade['feasible']}")
feasible_results = [r for r in trade_results if r['feasible']]
if len(feasible_results) > 0:
    best_feasible = max(feasible_results, key=lambda r: r['endurance'])
    print()
    print('BEST FEASIBLE ENDURANCE DESIGN')
    print('-' * 72)
    print(f"Wing area:           {best_feasible['S']:.2f} m^2")
    print(f"Aspect ratio:        {best_feasible['AR']:.1f}")
    print(f"Endurance:           {best_feasible['endurance']:.1f} min")
    print(f"Maximum load factor: {best_feasible['n_max']:.2f}")
else:
    print()
    print('No design satisfies all CG, structural, and flight feasibility checks.')
plane.despmtr['wing_area'].value = 0.6
plane.despmtr['wing_AR'].value = 8.0
