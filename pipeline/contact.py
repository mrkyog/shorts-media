import sys, os
from PIL import Image, ImageDraw, ImageFont
src, out = sys.argv[1], sys.argv[2]
files = sorted(f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.png')))
cols, cw, ch = 4, 480, 400
rows = (len(files) + cols - 1) // cols
sheet = Image.new('RGB', (cols * cw, rows * ch), (20, 20, 24))
d = ImageDraw.Draw(sheet)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
for i, f in enumerate(files):
    im = Image.open(os.path.join(src, f)).convert('RGB')
    w, h = im.size
    im.thumbnail((cw - 16, ch - 44))
    x, y = (i % cols) * cw, (i // cols) * ch
    sheet.paste(im, (x + 8, y + 8))
    d.text((x + 8, y + ch - 32), f"{f[:28]}  {w}x{h}", fill=(255, 255, 255), font=font)
sheet.save(out, quality=85)
print(out, sheet.size, len(files))
