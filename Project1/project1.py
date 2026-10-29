from pathlib import Path
import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
import pyCAPS

# ============================================================
# MAE 451 PROJECT 1
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
CSM_FILE = PROJECT_DIR / "aircraft.csm"
CAPS_RUN_DIR = PROJECT_DIR / "caps_run"

FIGURE_DIR = PROJECT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

os.chdir(PROJECT_DIR)

if CAPS_RUN_DIR.exists():
    shutil.rmtree(CAPS_RUN_DIR)

caps_problem = pyCAPS.Problem(
    problemName=str(CAPS_RUN_DIR),
    capsFile=str(CSM_FILE),
    outLevel=0
)

plane = caps_problem.geometry


# ============================================================
# CONSTANTS
# ============================================================

g = 9.80665

rho_sl = 1.225        # kg/m^3
mu_sl = 1.789e-5      # kg/(m*s)

rho_foam = 30.0       # kg/m^3

# Hardware masses
motor_mass = 0.080
prop_mass = 0.0190
esc_mass = 0.082
battery_mass = 0.348
receiver_mass = 0.0145
servo_mass = 0.0166

# Aerodynamic assumptions
oswald_efficiency = 0.80

# Simplified profile-drag assumptions
# Replace later with XFOIL/AirfoilTools data if desired
cd_wing_profile = 0.010
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

    # Geometry
    wing_x = despmtr("wing_x")
    wing_chord = outpmtr("wing_chord")

    # Foam geometry
    nose_volume = outpmtr("noseV")
    nose_xcg = outpmtr("nosecg")

    fuse_volume = outpmtr("fuseV")
    fuse_xcg = outpmtr("fusecg")

    wing_volume = outpmtr("wingV")
    wing_xcg = outpmtr("wingcg")

    htail_volume = outpmtr("htailV")
    htail_xcg = outpmtr("htailcg")

    vtail_volume = outpmtr("vtailV")
    vtail_xcg = outpmtr("vtailcg")

    # Hardware locations
    motor_xcg = outpmtr("motorcg")
    prop_xcg = outpmtr("propcg")

    esc_xcg = outpmtr("esccg")
    battery_xcg = outpmtr("battcg")
    receiver_xcg = outpmtr("rxcg")

    ail_left_xcg = outpmtr("ail_left_cg")
    ail_right_xcg = outpmtr("ail_right_cg")
    elevator_xcg = outpmtr("elev_cg")
    rudder_xcg = outpmtr("rud_cg")

    # Hardware volumes
    esc_volume = outpmtr("escV")
    battery_volume = outpmtr("battV")
    receiver_volume = outpmtr("rxV")
    servo_volume = outpmtr("servoV")

    parts = []

    def add(name, mass, x):
        parts.append((name, mass, x))

    # Foam
    add("Nose foam", rho_foam*nose_volume, nose_xcg)
    add("Fuselage foam", rho_foam*fuse_volume, fuse_xcg)
    add("Main wing foam", rho_foam*wing_volume, wing_xcg)
    add("Horizontal tail foam", rho_foam*htail_volume, htail_xcg)
    add("Vertical tail foam", rho_foam*vtail_volume, vtail_xcg)

    # Hardware
    add("Motor", motor_mass, motor_xcg)
    add("Propeller", prop_mass, prop_xcg)
    add("ESC", esc_mass, esc_xcg)
    add("Battery", battery_mass, battery_xcg)
    add("Receiver", receiver_mass, receiver_xcg)

    add("Left aileron servo", servo_mass, ail_left_xcg)
    add("Right aileron servo", servo_mass, ail_right_xcg)
    add("Elevator servo", servo_mass, elevator_xcg)
    add("Rudder servo", servo_mass, rudder_xcg)

    # Displaced foam
    add("ESC foam removed", -rho_foam*esc_volume, esc_xcg)
    add("Battery foam removed", -rho_foam*battery_volume, battery_xcg)
    add("Receiver foam removed", -rho_foam*receiver_volume, receiver_xcg)

    add("Left servo foam removed",
        -rho_foam*servo_volume, ail_left_xcg)

    add("Right servo foam removed",
        -rho_foam*servo_volume, ail_right_xcg)

    add("Elevator foam removed",
        -rho_foam*servo_volume, elevator_xcg)

    add("Rudder foam removed",
        -rho_foam*servo_volume, rudder_xcg)

    total_mass = sum(p[1] for p in parts)

    total_moment = sum(
        p[1] * p[2]
        for p in parts
    )

    xcg = total_moment / total_mass

    weight = total_mass * g

    cg_fraction = (
        (xcg - wing_x)
        / wing_chord
    )

    return {
        "mass": total_mass,
        "weight": weight,
        "xcg": xcg,
        "cg_fraction": cg_fraction,
        "cg_percent": 100*cg_fraction,
        "parts": parts
    }


# ============================================================
# AERODYNAMIC MODEL
# STEP 5
# ============================================================

def aero_model(V, weight, rho=rho_sl):

    V = np.asarray(V)

    # Aircraft geometry
    S = despmtr("wing_area")
    AR = despmtr("wing_AR")

    fus_length = despmtr("fus_length")
    fus_diameter = outpmtr("fus_diameter")

    ht_area = outpmtr("ht_area")
    vt_area = outpmtr("vt_area")

    # Dynamic pressure
    q = 0.5 * rho * V**2

    # --------------------------------------------------------
    # LIFT
    #
    # Steady level flight:
    # L = W
    #
    # L = q S CL
    # --------------------------------------------------------

    CL = weight / (q*S)

    lift = q*S*CL

    # --------------------------------------------------------
    # WING PROFILE DRAG
    #
    # Assignment says wing profile coefficients may be
    # represented by NACA 2412 data.
    #
    # For this preliminary model, use constant profile CD.
    # --------------------------------------------------------

    CD_wing = (
        cd_wing_profile
        * np.ones_like(V)
    )

    # --------------------------------------------------------
    # FUSELAGE DRAG
    #
    # Lecture 6:
    #
    # CD_fuse =
    # (L*d/S) [
    # 0.25(L/d)^-0.9
    # + 0.05(L/d)^0.57
    # ]
    # --------------------------------------------------------

    fineness_ratio = (
        fus_length
        / fus_diameter
    )

    CD_fuse = (
        fus_length
        * fus_diameter
        / S
    ) * (
        0.25
        * fineness_ratio**(-0.9)
        +
        0.05
        * fineness_ratio**0.57
    )

    CD_fuse = (
        CD_fuse
        * np.ones_like(V)
    )

    # --------------------------------------------------------
    # TAIL PROFILE DRAG
    #
    # Tail CL assumed zero per assignment.
    # Tail coefficients referenced to wing area.
    # --------------------------------------------------------

    CD_ht = (
        cd_tail_profile
        * ht_area
        / S
    )

    CD_vt = (
        cd_tail_profile
        * vt_area
        / S
    )

    CD_ht = (
        CD_ht
        * np.ones_like(V)
    )

    CD_vt = (
        CD_vt
        * np.ones_like(V)
    )

    # --------------------------------------------------------
    # INDUCED DRAG
    # --------------------------------------------------------

    CD_induced = (
        CL**2
        /
        (
            np.pi
            * oswald_efficiency
            * AR
        )
    )

    # --------------------------------------------------------
    # TOTAL DRAG
    # --------------------------------------------------------

    CD_total = (
        CD_wing
        + CD_fuse
        + CD_ht
        + CD_vt
        + CD_induced
    )

    drag = (
        q
        * S
        * CD_total
    )

    return {
        "V": V,
        "q": q,
        "CL": CL,
        "lift": lift,

        "CD_wing": CD_wing,
        "CD_fuse": CD_fuse,
        "CD_ht": CD_ht,
        "CD_vt": CD_vt,
        "CD_induced": CD_induced,

        "CD_total": CD_total,
        "drag": drag
    }


# ============================================================
# PROPULSION MODEL
# ============================================================

def propulsion_model(
    V,
    throttle,
    rho=rho_sl
):

    V = np.asarray(V)

    # Assignment thrust model
    thrust = (
        15.0
        * (rho/rho_sl)
        * (1.0 - V/50.0)
        * throttle
    )

    # No negative thrust past model limit
    thrust = np.maximum(
        thrust,
        0.0
    )

    efficiency = 0.70

    return {
        "thrust": thrust,
        "efficiency": efficiency
    }


# ============================================================
# BASELINE AIRCRAFT ANALYSIS
# ============================================================

weights = weights_model()

print()
print("="*72)
print("BASELINE AIRCRAFT")
print("="*72)

print(
    f"Mass:             "
    f"{weights['mass']:.4f} kg"
)

print(
    f"Weight:           "
    f"{weights['weight']:.4f} N"
)

print(
    f"CG from nose:     "
    f"{weights['xcg']:.4f} m"
)

print(
    f"CG location:      "
    f"{weights['cg_percent']:.2f}% chord"
)


# ============================================================
# STEP 5 CHECK AT ONE VELOCITY
# ============================================================

V_test = 20.0

aero_test = aero_model(
    V_test,
    weights["weight"]
)

print()
print("="*72)
print("STEP 5 - AERODYNAMIC MODEL CHECK AT 20 m/s")
print("="*72)

print(
    f"CL:               "
    f"{float(aero_test['CL']):.4f}"
)

print(
    f"CD wing profile:  "
    f"{float(aero_test['CD_wing']):.4f}"
)

print(
    f"CD fuselage:      "
    f"{float(aero_test['CD_fuse']):.4f}"
)

print(
    f"CD horizontal:    "
    f"{float(aero_test['CD_ht']):.4f}"
)

print(
    f"CD vertical:      "
    f"{float(aero_test['CD_vt']):.4f}"
)

print(
    f"CD induced:       "
    f"{float(aero_test['CD_induced']):.4f}"
)

print(
    f"CD total:         "
    f"{float(aero_test['CD_total']):.4f}"
)

print(
    f"Drag:             "
    f"{float(aero_test['drag']):.4f} N"
)


# ============================================================
# STEP 6
# DRAG VS VELOCITY + THRUST CURVES
# ============================================================

V = np.linspace(
    4,
    49.0,
    400
)

aero = aero_model(
    V,
    weights["weight"]
)

T20 = propulsion_model(
    V,
    0.20
)["thrust"]

T50 = propulsion_model(
    V,
    0.50
)["thrust"]

T100 = propulsion_model(
    V,
    1.00
)["thrust"]


plt.figure(figsize=(8, 5))

plt.plot(
    V,
    aero["drag"],
    label="Drag Required"
)

plt.plot(
    V,
    T20,
    label="20% Throttle"
)

plt.plot(
    V,
    T50,
    label="50% Throttle"
)

plt.plot(
    V,
    T100,
    label="100% Throttle"
)

plt.xlabel(
    "Velocity (m/s)"
)

plt.ylabel(
    "Force (N)"
)

plt.xlim(4, 49)

plt.title(
    "Drag Required and Thrust Available at Sea Level"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "step6_drag_thrust.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# STEP 7 - PERFORMANCE
# ============================================================

# ------------------------------------------------------------
# BATTERY / AIRCRAFT ASSUMPTIONS
# ------------------------------------------------------------

CL_max = 1.50

battery_voltage = 14.8       # V, nominal 4S LiPo
battery_capacity_Ah = 3.3    # Ah

battery_energy_Wh = (
    battery_voltage
    * battery_capacity_Ah
)

battery_energy_J = (
    battery_energy_Wh
    * 3600.0
)

powertrain_efficiency = 0.70


# ============================================================
# STANDARD ATMOSPHERE
# ============================================================

def atmosphere(h):

    T0 = 288.15
    p0 = 101325.0
    rho0 = 1.225

    R = 287.05
    g0 = 9.80665

    # Troposphere
    if h <= 11000.0:

        L = 0.0065

        T = T0 - L*h

        p = (
            p0
            * (T/T0)**(g0/(R*L))
        )

    # Lower stratosphere
    else:

        T11 = 216.65

        p11 = (
            p0
            * (T11/T0)**(g0/(R*0.0065))
        )

        T = T11

        p = (
            p11
            * np.exp(
                -g0*(h-11000.0)
                /(R*T)
            )
        )

    rho = p/(R*T)

    return rho


# ============================================================
# SEA-LEVEL PERFORMANCE GRID
# ============================================================

W = weights["weight"]
S = despmtr("wing_area")

V_stall_SL = np.sqrt(
    2.0*W
    /
    (
        rho_sl
        * S
        * CL_max
    )
)

V_perf = np.linspace(
    V_stall_SL,
    49.0,
    3000
)

aero_perf = aero_model(
    V_perf,
    W,
    rho_sl
)

D_perf = aero_perf["drag"]

P_flight = (
    D_perf
    * V_perf
)

P_battery = (
    P_flight
    / powertrain_efficiency
)


# ============================================================
# BEST RANGE
#
# Electric aircraft:
#
# Range = eta * E_battery / D
#
# therefore maximum range occurs at minimum drag.
# ============================================================

i_range = np.argmin(
    D_perf
)

V_best_range = (
    V_perf[i_range]
)

D_best_range = (
    D_perf[i_range]
)

best_range = (
    powertrain_efficiency
    * battery_energy_J
    / D_best_range
)


# ============================================================
# BEST ENDURANCE
#
# Endurance = eta * E_battery / (D*V)
#
# therefore maximize endurance by minimizing required power.
# ============================================================

i_endurance = np.argmin(
    P_flight
)

V_best_endurance = (
    V_perf[i_endurance]
)

P_min_flight = (
    P_flight[i_endurance]
)

P_min_battery = (
    P_battery[i_endurance]
)

best_endurance_sec = (
    battery_energy_J
    / P_min_battery
)

best_endurance_min = (
    best_endurance_sec
    / 60.0
)


# ============================================================
# CEILING
#
# Rate of climb:
#
# ROC = excess power / W
#     = (T-D)*V/W
#
# Absolute ceiling:
# max ROC = 0
#
# Service ceiling:
# max ROC = 100 ft/min = 0.508 m/s
# ============================================================

altitudes = np.linspace(
    0.0,
    20000.0,
    1001
)

max_ROC = np.zeros_like(
    altitudes
)

best_climb_speed = np.zeros_like(
    altitudes
)

for i, h in enumerate(altitudes):

    rho = atmosphere(h)

    V_stall_h = np.sqrt(
        2.0*W
        /
        (
            rho
            * S
            * CL_max
        )
    )

    V_h = np.linspace(
        V_stall_h,
        49.0,
        800
    )

    aero_h = aero_model(
        V_h,
        W,
        rho
    )

    thrust_h = propulsion_model(
        V_h,
        1.0,
        rho
    )["thrust"]

    ROC = (
        (thrust_h-aero_h["drag"])
        * V_h
        / W
    )

    i_max = np.argmax(
        ROC
    )

    max_ROC[i] = (
        ROC[i_max]
    )

    best_climb_speed[i] = (
        V_h[i_max]
    )


# Absolute ceiling
valid_absolute = np.where(
    max_ROC >= 0.0
)[0]

if len(valid_absolute) > 0:

    absolute_ceiling = (
        altitudes[
            valid_absolute[-1]
        ]
    )

else:

    absolute_ceiling = np.nan


# Service ceiling
service_ROC = 0.508  # m/s = 100 ft/min

valid_service = np.where(
    max_ROC >= service_ROC
)[0]

if len(valid_service) > 0:

    service_ceiling = (
        altitudes[
            valid_service[-1]
        ]
    )

else:

    service_ceiling = np.nan


# ============================================================
# STALL SPEED VS BANK ANGLE
#
# n = 1/cos(phi)
#
# Vs(phi) =
# sqrt(
#   2 n W /
#   rho S CLmax
# )
# ============================================================

bank_deg = np.linspace(
    0.0,
    80.0,
    161
)

bank_rad = np.deg2rad(
    bank_deg
)

load_factor = (
    1.0
    / np.cos(bank_rad)
)

V_stall_bank = np.sqrt(
    2.0
    * load_factor
    * W
    /
    (
        rho_sl
        * S
        * CL_max
    )
)


# ============================================================
# DASH SPEED VS BANK ANGLE
#
# Level turn requires:
#
# L = nW
#
# For each bank angle, find highest velocity where
#
# T_available >= D_required
# ============================================================

V_dash_bank = np.full_like(
    bank_deg,
    np.nan
)

for i, n in enumerate(
    load_factor
):

    V_min = (
        V_stall_bank[i]
    )

    V_test = np.linspace(
        V_min,
        49.0,
        2000
    )

    # Passing n*W makes aero_model compute
    # lift coefficient required for the turn
    aero_turn = aero_model(
        V_test,
        n*W,
        rho_sl
    )

    T_turn = propulsion_model(
        V_test,
        1.0,
        rho_sl
    )["thrust"]

    feasible = np.where(
        T_turn
        >= aero_turn["drag"]
    )[0]

    if len(feasible) > 0:

        V_dash_bank[i] = (
            V_test[
                feasible[-1]
            ]
        )


# ============================================================
# MINIMUM TURN RADIUS
#
# R = V^2 / (g tan(phi))
#
# At each bank angle, use stall-limited minimum velocity.
# Also require thrust sufficient to maintain level turn.
# ============================================================

turn_radius = np.full_like(
    bank_deg,
    np.nan
)

for i, phi in enumerate(
    bank_rad
):

    if phi == 0.0:
        continue

    V_turn = (
        V_stall_bank[i]
    )

    n = (
        load_factor[i]
    )

    aero_turn = aero_model(
        V_turn,
        n*W,
        rho_sl
    )

    thrust_turn = propulsion_model(
        V_turn,
        1.0,
        rho_sl
    )["thrust"]

    # Only count sustained turns
    if (
        thrust_turn
        >= aero_turn["drag"]
    ):

        turn_radius[i] = (
            V_turn**2
            /
            (
                g
                * np.tan(phi)
            )
        )


if np.any(
    np.isfinite(turn_radius)
):

    i_min_radius = np.nanargmin(
        turn_radius
    )

    minimum_turn_radius = (
        turn_radius[
            i_min_radius
        ]
    )

    min_radius_bank = (
        bank_deg[
            i_min_radius
        ]
    )

    min_radius_speed = (
        V_stall_bank[
            i_min_radius
        ]
    )

else:

    minimum_turn_radius = np.nan
    min_radius_bank = np.nan
    min_radius_speed = np.nan


# ============================================================
# STEP 7 OUTPUT
# ============================================================

print()
print("="*72)
print("STEP 7 - PERFORMANCE")
print("="*72)

print()
print("BEST RANGE")
print("-"*72)

print(
    f"Best range speed:       "
    f"{V_best_range:.3f} m/s"
)

print(
    f"Minimum drag:           "
    f"{D_best_range:.3f} N"
)

print(
    f"Best range:             "
    f"{best_range/1000:.3f} km"
)


print()
print("BEST ENDURANCE")
print("-"*72)

print(
    f"Best endurance speed:   "
    f"{V_best_endurance:.3f} m/s"
)

print(
    f"Minimum flight power:   "
    f"{P_min_flight:.3f} W"
)

print(
    f"Battery power required: "
    f"{P_min_battery:.3f} W"
)

print(
    f"Best endurance:         "
    f"{best_endurance_min:.2f} min"
)


print()
print("CEILINGS")
print("-"*72)

print(
    f"Service ceiling:        "
    f"{service_ceiling:.0f} m"
)

print(
    f"Absolute ceiling:       "
    f"{absolute_ceiling:.0f} m"
)


print()
print("STALL")
print("-"*72)

print(
    f"Sea-level stall speed:  "
    f"{V_stall_SL:.3f} m/s"
)


print()
print("MINIMUM TURN RADIUS")
print("-"*72)

print(
    f"Minimum turn radius:    "
    f"{minimum_turn_radius:.3f} m"
)

print(
    f"Bank angle:             "
    f"{min_radius_bank:.1f} deg"
)

print(
    f"Turn speed:             "
    f"{min_radius_speed:.3f} m/s"
)


# ============================================================
# STEP 7 FIGURE 1
# STALL SPEED VS BANK ANGLE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    bank_deg,
    V_stall_bank
)

plt.xlabel(
    "Bank Angle (deg)"
)

plt.ylabel(
    "Stall Speed (m/s)"
)

plt.title(
    "Stall Speed vs Bank Angle"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "step7_stall_speed_vs_bank.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 7 FIGURE 2
# DASH SPEED VS BANK ANGLE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    bank_deg,
    V_dash_bank
)

plt.xlabel(
    "Bank Angle (deg)"
)

plt.ylabel(
    "Dash Speed (m/s)"
)

plt.title(
    "Dash Speed vs Bank Angle"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "step7_dash_speed_vs_bank.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STEP 7 FIGURE 3
# MAXIMUM RATE OF CLIMB VS ALTITUDE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    altitudes,
    max_ROC
)

plt.axhline(
    0.508,
    linestyle="--",
    label="Service Ceiling Criterion"
)

plt.axhline(
    0.0,
    linestyle="--",
    label="Absolute Ceiling Criterion"
)

plt.xlabel(
    "Altitude (m)"
)

plt.ylabel(
    "Maximum Rate of Climb (m/s)"
)

plt.title(
    "Maximum Rate of Climb vs Altitude"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "step7_ceiling.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# STEP 8 - WING STRUCTURAL LIMIT
# ============================================================

# Foam ultimate strengths
sigma_tensile = 140e3       # Pa
sigma_compressive = 100e3   # Pa

# Lower bending allowable governs
sigma_allow = min(
    sigma_tensile,
    sigma_compressive
)

# NACA 2412 normalized bending moment of inertia
I_norm_2412 = 6.84e-5

# Wing geometry
b = outpmtr("wing_span")
c_wing = outpmtr("wing_chord")

# Actual section moment of inertia
I_wing = (
    I_norm_2412
    * c_wing**4
)

# Maximum distance from neutral axis
# NACA 2412 is 12% thick -> half thickness = 0.06c
c_bending = (
    0.06
    * c_wing
)

# Elliptical half-wing lift resultant location
y_lift = (
    4.0*b
    / (6.0*np.pi)
)

# ------------------------------------------------------------
# ROOT BENDING MOMENT
#
# Total lift in a level turn:
# L = nW
#
# One wing carries:
# L_half = nW/2
#
# Root moment:
# M_root = (nW/2)*y_lift
#
# Bending stress:
# sigma = M*c/I
# ------------------------------------------------------------

root_moment_1g = (
    (W/2.0)
    * y_lift
)

stress_1g = (
    root_moment_1g
    * c_bending
    / I_wing
)

# Solve sigma_allow = n * stress_1g
n_structural_max = (
    sigma_allow
    / stress_1g
)


# ------------------------------------------------------------
# MAXIMUM BANK ANGLE
#
# Level coordinated turn:
#
# n = 1/cos(phi)
# ------------------------------------------------------------

if n_structural_max >= 1.0:

    phi_structural_max = (
        np.degrees(
            np.arccos(
                1.0/n_structural_max
            )
        )
    )

    structural_feasible = True

else:

    phi_structural_max = np.nan
    structural_feasible = False


# ============================================================
# STEP 8 OUTPUT
# ============================================================

print()
print("="*72)
print("STEP 8 - WING STRUCTURAL LIMIT")
print("="*72)

print(
    f"Wing span:                   "
    f"{b:.4f} m"
)

print(
    f"Wing chord:                  "
    f"{c_wing:.4f} m"
)

print(
    f"Bending moment of inertia:   "
    f"{I_wing:.6e} m^4"
)

print(
    f"Half-wing lift arm:          "
    f"{y_lift:.4f} m"
)

print(
    f"1-g root bending moment:     "
    f"{root_moment_1g:.4f} N*m"
)

print(
    f"1-g bending stress:          "
    f"{stress_1g/1000:.2f} kPa"
)

print(
    f"Allowable bending stress:    "
    f"{sigma_allow/1000:.2f} kPa"
)

print(
    f"Maximum structural load n:   "
    f"{n_structural_max:.3f}"
)

if structural_feasible:

    print(
        f"Maximum structural bank:     "
        f"{phi_structural_max:.2f} deg"
    )

else:

    print(
        "Maximum structural bank:     N/A"
    )

    print(
        "STRUCTURAL CHECK:             FAILS BELOW 1 g"
    )

    print(
        "The baseline wing cannot support "
        "steady level flight under the stated "
        "foam properties and beam model."
    )

