import sys, os
from PIL import Image
src, out, cols, tw = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
files = sorted(f for f in os.listdir(src) if f.endswith('.jpg'))
if len(sys.argv) > 5:
    want = set(sys.argv[5].split(','))
    files = [f for f in files if f[2:6] in want]
th = tw * 16 // 9
rows = (len(files) + cols - 1) // cols
sheet = Image.new('RGB', (cols * (tw + 6) + 6, rows * (th + 6) + 6), (60, 60, 66))
for i, f in enumerate(files):
    im = Image.open(os.path.join(src, f)).convert('RGB').resize((tw, th), Image.LANCZOS)
    sheet.paste(im, (6 + (i % cols) * (tw + 6), 6 + (i // cols) * (th + 6)))
sheet.save(out, quality=88)
print(out, sheet.size, files)
