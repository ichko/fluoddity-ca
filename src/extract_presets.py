"""Rebuild presets.js from a clone of Fluoddity (github.com/aphid91/Fluoddity).

Reads physics_configs/Core and physics_configs/Advanced, fills in what an old
file leaves out with the defaults Fluoddity's own loader uses, and writes one
flat object per preset with the 80-number rule.

    python3 src/extract_presets.py path/to/Fluoddity
"""
import glob, json, os, sys
src = sys.argv[1] if len(sys.argv) > 1 else "Fluoddity"
D = dict(axial_force=0.371, lateral_force=-0.707, sensor_gain=0.116, mutation_scale=0.0, drag=0.504,
         strafe_power=0.224, sensor_angle=0.45, global_force_mult=1.0, sensor_distance=1.0,
         trail_persistence=0.938, trail_diffusion=1.0, hazard_rate=0.0)
S = dict(disable_symmetry=False, absolute_orientation=0, orientation_mix=1.0, boundary_conditions=0,
         initial_conditions=0, num_cohorts=64, rule_seed=0.42)
A = dict(ink_weight=1.0, hue_sensitivity=0.5, color_by_cohort=True, watercolor_mode=False)
r = lambda x: float("%.7g" % x)
out = []
for f in sorted(glob.glob(os.path.join(src, "physics_configs/Core/*.json"))) + \
         sorted(glob.glob(os.path.join(src, "physics_configs/Advanced/*.json"))):
    d = json.load(open(f))
    c = {"name": os.path.basename(f)[:-5], "group": os.path.basename(os.path.dirname(f))}
    for k, v in D.items(): c[k] = r(d["physics"].get(k, v))
    for k, v in S.items():
        v2 = d["settings"].get(k, v)
        c[k] = v2 if isinstance(v2, bool) else (r(v2) if isinstance(v2, float) else int(v2))
    for k, v in A.items():
        v2 = d["appearance"].get(k, v)
        c[k] = r(v2) if isinstance(v2, float) else v2
    c["rule"] = [r(x) for x in d.get("rule", [0.0]*80)]
    out.append(c)
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.js"), "w").write(
    "const PRESETS = " + json.dumps(out, separators=(",", ":")) + ";\n")
print(len(out), "presets")
