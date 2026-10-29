from pathlib import Path
import shutil

import numpy as np
import pyCAPS


# ============================================================
# MAE 451 PROJECT 1
# STEP 3 + STEP 4
#
# Geometry source of truth:
#     aircraft.csm
#
# Python handles:
#     - component masses
#     - foam mass
#     - foam displacement
#     - aircraft CG
#     - CG requirement
#     - recommended wing position
# ============================================================


# ============================================================
# FILE LOCATIONS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CSM_FILE = PROJECT_DIR / "aircraft.csm"

CAPS_RUN_DIR = PROJECT_DIR / "caps_run"


if not CSM_FILE.exists():
    raise FileNotFoundError(
        f"Could not find aircraft.csm at:\n{CSM_FILE}"
    )


# Delete old pyCAPS working directory so script can be rerun cleanly
if CAPS_RUN_DIR.exists():
    shutil.rmtree(CAPS_RUN_DIR)


# ============================================================
# LOAD OPENCSM MODEL THROUGH pyCAPS
# ============================================================

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

rho_foam = 30.0  # kg/m^3


# ============================================================
# HARDWARE MASSES
# kg
# ============================================================

motor_mass = 0.080
prop_mass = 0.0190

esc_mass = 0.082
battery_mass = 0.348
receiver_mass = 0.0145

servo_mass = 0.0166


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def despmtr(name):
    """Return an OpenCSM design parameter."""
    return plane.despmtr[name].value


def outpmtr(name):
    """Return an OpenCSM output parameter."""
    return plane.outpmtr[name].value


def add_part(parts, name, mass, xcg):
    """
    Add one mass contribution to the CG table.
    mass may be negative for displaced foam.
    """
    parts.append(
        {
            "name": name,
            "mass": float(mass),
            "xcg": float(xcg),
            "moment": float(mass * xcg),
        }
    )


# ============================================================
# READ CURRENT DESIGN PARAMETERS FROM aircraft.csm
# ============================================================

wing_area = despmtr("wing_area")
wing_AR = despmtr("wing_AR")

fus_length = despmtr("fus_length")
fus_FR = despmtr("fus_FR")

wing_x = despmtr("wing_x")


# ============================================================
# READ DERIVED GEOMETRY DIRECTLY FROM aircraft.csm
# ============================================================

wing_span = outpmtr("wing_span")
wing_chord = outpmtr("wing_chord")

fus_diameter = outpmtr("fus_diameter")
fus_radius = outpmtr("fus_radius")

ht_area = outpmtr("ht_area")
ht_span = outpmtr("ht_span")
ht_chord = outpmtr("ht_chord")

vt_area = outpmtr("vt_area")
vt_height = outpmtr("vt_height")


# ============================================================
# READ ACTUAL FOAM BODY VOLUMES + CENTROIDS FROM OPENCSM
# ============================================================

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


# ============================================================
# FOAM MASSES
# ============================================================

nose_mass = rho_foam * nose_volume
fuse_mass = rho_foam * fuse_volume

wing_mass = rho_foam * wing_volume

htail_mass = rho_foam * htail_volume
vtail_mass = rho_foam * vtail_volume


# ============================================================
# READ ACTUAL COMPONENT LOCATIONS FROM OPENCSM
# ============================================================

motor_xcg = outpmtr("motorcg")
prop_xcg = outpmtr("propcg")

esc_xcg = outpmtr("esccg")
battery_xcg = outpmtr("battcg")
receiver_xcg = outpmtr("rxcg")

ail_left_xcg = outpmtr("ail_left_cg")
ail_right_xcg = outpmtr("ail_right_cg")

elevator_xcg = outpmtr("elev_cg")
rudder_xcg = outpmtr("rud_cg")


# ============================================================
# ACTUAL COMPONENT VOLUMES FROM OPENCSM
# Used to calculate displaced foam
# ============================================================

esc_volume = outpmtr("escV")
battery_volume = outpmtr("battV")
receiver_volume = outpmtr("rxV")

servo_volume = outpmtr("servoV")


# ============================================================
# FOAM DISPLACEMENT
# ============================================================

esc_foam_removed = rho_foam * esc_volume
battery_foam_removed = rho_foam * battery_volume
receiver_foam_removed = rho_foam * receiver_volume

servo_foam_removed = rho_foam * servo_volume


# ============================================================
# BUILD COMPONENT MASS TABLE
# ============================================================

parts = []


# -------------------------
# Foam
# -------------------------

add_part(
    parts,
    "Nose foam",
    nose_mass,
    nose_xcg
)

add_part(
    parts,
    "Fuselage foam",
    fuse_mass,
    fuse_xcg
)

add_part(
    parts,
    "Main wing foam",
    wing_mass,
    wing_xcg
)

add_part(
    parts,
    "Horizontal tail foam",
    htail_mass,
    htail_xcg
)

add_part(
    parts,
    "Vertical tail foam",
    vtail_mass,
    vtail_xcg
)


# -------------------------
# Hardware
# -------------------------

add_part(
    parts,
    "Motor",
    motor_mass,
    motor_xcg
)

add_part(
    parts,
    "Propeller",
    prop_mass,
    prop_xcg
)

add_part(
    parts,
    "ESC",
    esc_mass,
    esc_xcg
)

add_part(
    parts,
    "Battery",
    battery_mass,
    battery_xcg
)

add_part(
    parts,
    "Receiver",
    receiver_mass,
    receiver_xcg
)

add_part(
    parts,
    "Left aileron servo",
    servo_mass,
    ail_left_xcg
)

add_part(
    parts,
    "Right aileron servo",
    servo_mass,
    ail_right_xcg
)

add_part(
    parts,
    "Elevator servo",
    servo_mass,
    elevator_xcg
)

add_part(
    parts,
    "Rudder servo",
    servo_mass,
    rudder_xcg
)


# -------------------------
# Foam displaced by electronics
# -------------------------

add_part(
    parts,
    "ESC foam removed",
    -esc_foam_removed,
    esc_xcg
)

add_part(
    parts,
    "Battery foam removed",
    -battery_foam_removed,
    battery_xcg
)

add_part(
    parts,
    "Receiver foam removed",
    -receiver_foam_removed,
    receiver_xcg
)

add_part(
    parts,
    "Left aileron foam removed",
    -servo_foam_removed,
    ail_left_xcg
)

add_part(
    parts,
    "Right aileron foam removed",
    -servo_foam_removed,
    ail_right_xcg
)

add_part(
    parts,
    "Elevator foam removed",
    -servo_foam_removed,
    elevator_xcg
)

add_part(
    parts,
    "Rudder foam removed",
    -servo_foam_removed,
    rudder_xcg
)


# ============================================================
# TOTAL MASS + CG
# ============================================================

total_mass = sum(
    part["mass"]
    for part in parts
)

total_moment = sum(
    part["moment"]
    for part in parts
)

aircraft_xcg = (
    total_moment
    / total_mass
)

aircraft_weight = (
    total_mass
    * g
)


# ============================================================
# CG RELATIVE TO MAIN WING
# ============================================================

cg_fraction = (
    (aircraft_xcg - wing_x)
    / wing_chord
)

cg_percent = (
    100.0
    * cg_fraction
)

cg_min_x = (
    wing_x
    + 0.25 * wing_chord
)

cg_max_x = (
    wing_x
    + 0.50 * wing_chord
)

cg_pass = (
    0.25 <= cg_fraction <= 0.50
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 82)
print("MAE 451 PROJECT 1")
print("OPENCSM / pyCAPS WEIGHT AND CG ANALYSIS")
print("=" * 82)


print()
print("GEOMETRY READ DIRECTLY FROM aircraft.csm")
print("-" * 82)

print(
    f"Wing area:                  "
    f"{wing_area:.6f} m^2"
)

print(
    f"Wing aspect ratio:          "
    f"{wing_AR:.6f}"
)

print(
    f"Wing span:                  "
    f"{wing_span:.6f} m"
)

print(
    f"Wing chord:                 "
    f"{wing_chord:.6f} m"
)

print(
    f"Wing leading edge:          "
    f"{wing_x:.6f} m"
)

print(
    f"Fuselage length:            "
    f"{fus_length:.6f} m"
)

print(
    f"Fuselage diameter:          "
    f"{fus_diameter:.6f} m"
)

print(
    f"Horizontal tail area:       "
    f"{ht_area:.6f} m^2"
)

print(
    f"Vertical tail area:         "
    f"{vt_area:.6f} m^2"
)


print()
print("OPENCSM FOAM GEOMETRY")
print("-" * 82)

print(
    f"{'Component':25s}"
    f"{'Volume (m^3)':>18s}"
    f"{'Foam Mass (kg)':>18s}"
    f"{'x_CG (m)':>15s}"
)

print("-" * 82)

foam_rows = [
    (
        "Nose",
        nose_volume,
        nose_mass,
        nose_xcg
    ),
    (
        "Fuselage",
        fuse_volume,
        fuse_mass,
        fuse_xcg
    ),
    (
        "Main wing",
        wing_volume,
        wing_mass,
        wing_xcg
    ),
    (
        "Horizontal tail",
        htail_volume,
        htail_mass,
        htail_xcg
    ),
    (
        "Vertical tail",
        vtail_volume,
        vtail_mass,
        vtail_xcg
    ),
]

for name, volume, mass, xcg in foam_rows:

    print(
        f"{name:25s}"
        f"{volume:18.8f}"
        f"{mass:18.6f}"
        f"{xcg:15.6f}"
    )


print()
print("COMPONENT MASS / MOMENT TABLE")
print("-" * 82)

print(
    f"{'Component':32s}"
    f"{'Mass (kg)':>14s}"
    f"{'x_CG (m)':>14s}"
    f"{'Moment':>16s}"
)

print("-" * 82)

for part in parts:

    print(
        f"{part['name']:32s}"
        f"{part['mass']:14.6f}"
        f"{part['xcg']:14.6f}"
        f"{part['moment']:16.8f}"
    )


print()
print("STEP 3 - CENTER OF GRAVITY")
print("-" * 82)

print(
    f"Aircraft CG from nose:      "
    f"{aircraft_xcg:.6f} m"
)

print(
    f"CG location:                "
    f"{cg_percent:.2f}% chord"
)

print(
    f"Allowed CG range:           "
    f"25% to 50% chord"
)

print(
    f"Allowed x_CG range:         "
    f"{cg_min_x:.6f} to "
    f"{cg_max_x:.6f} m"
)

if cg_pass:

    print(
        "CG CHECK:                   PASS"
    )

else:

    print(
        "CG CHECK:                   FAIL"
    )


print()
print("STEP 4 - AIRCRAFT WEIGHT")
print("-" * 82)

print(
    f"Total aircraft mass:        "
    f"{total_mass:.6f} kg"
)

print(
    f"Total aircraft weight:      "
    f"{aircraft_weight:.6f} N"
)

print(
    f"Total aircraft weight:      "
    f"{total_mass * 2.2046226218:.4f} lbm"
)


print()
print("=" * 82)