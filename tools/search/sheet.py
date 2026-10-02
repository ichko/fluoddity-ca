import sys, json
from PIL import Image, ImageDraw, ImageFont
# python3 sheet.py out.png cols "label|path" ...
out, cols = sys.argv[1], int(sys.argv[2]); items = [a.split('|', 1) for a in sys.argv[3:]]
S = 300; rows = (len(items) + cols - 1)//cols
im = Image.new('RGB', (cols*S, rows*S), (3, 5, 6)); d = ImageDraw.Draw(im)
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
except Exception: font = ImageFont.load_default()
for i, (lab, p) in enumerate(items):
    t = Image.open(p).convert('RGB').resize((S, S)); x, y = (i % cols)*S, (i//cols)*S
    im.paste(t, (x, y)); d.rectangle([x, y, x + 8 + 9*len(lab), y + 24], fill=(0, 0, 0)); d.text((x + 5, y + 3), lab, fill=(235, 235, 235), font=font)
im.save(out); print(out, im.size)
