"""Build the three pages: each template with the line /*PRESETS*/ replaced by presets.js.

    python3 src/build.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
presets = open(os.path.join(HERE, "presets.js")).read()
for name in ["fluoddity", "fluoddity-ca", "fluoddity-ca-next"]:
    t = open(os.path.join(HERE, name + ".tmpl.html")).read()
    open(os.path.join(ROOT, name + ".html"), "w").write(t.replace("/*PRESETS*/", presets))
    print("wrote", name + ".html")
