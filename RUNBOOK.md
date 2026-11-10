# Runbook: one short, end to end

Each scheduled run makes ONE vertical short and schedules it on Metricool. The
channel is English-language: mysteries of the universe/world, fun facts, "what if"
ideas, and series that continue across episodes.

## 0. Setup (fresh container every run)

```bash
# repo: if /home/claude/shorts-media is missing, attach mrkyog/shorts-media (push access) and clone it there
cd /home/claude/shorts-media && git pull -q
pip install --break-system-packages -q kokoro-onnx soundfile faster-whisper
mkdir -p ~/work/{models,assets,build,out} && cd ~/work/models
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  curl -sSL --fail --retry 3 -o $f.part https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f && mv $f.part $f
done
```
Fonts (Inter) are already in /usr/share/fonts/opentype/inter. ffmpeg, OpenCV, PIL and scipy are preinstalled.

## 1. Topic
- Read `topics.md`. Never repeat a topic. If the previous episode ended with "Follow for part 2: …", make that part 2 next.
- Search the web for space/science news of the last few days; prefer a real, datable hook. Otherwise pick an evergreen mystery or "what if".
- Every number in the script must come from a primary source (NASA, ESA, JPL, a journal, a university). Note the URLs.

## 2. Script → `episodes/YYYY-MM-DD-<slot>-<slug>.json`
Copy the structure of `episodes/2026-11-10-voyager-light-day.json`.
- 35–45 s, 9–11 lines. Line 1 is the hook and must make sense on a muted first frame.
- `/` splits caption groups (2–4 words each). `{spoken|shown}` when speech and caption differ (numbers, "Voyager {One.|1.}").
- Last line teases the next episode when there is a natural sequel.
- Voice `af_heart`, speed 1.0.

## 3. Pictures
- NASA image library (public domain; credit NASA): `https://images-api.nasa.gov/search?q=...&media_type=image`, then `/asset/<nasa_id>` for file URLs. Download `~orig.jpg`/`~large.jpg` into ~/work/assets.
- Build a contact sheet with `pipeline/contact.py` and LOOK at it before choosing. No logos, no people's faces as the subject, nothing copyrighted.
- Put the chosen files in the episode's `assets` map; write `shots` (`image` shots with from/to framing and stat `tags`).
- `race` and `scale` kinds in render.py are hard-wired to the Voyager script (they look up words in L6/L9). For a new infographic, add a new function + KINDS entry; don't edit the old ones.

## 4. Voice, frames, render
```bash
cd ~/work && P=/home/claude/shorts-media/pipeline E=/home/claude/shorts-media/episodes/<file>.json
python3 -I $P/tts_align.py $E models build          # every line must report match > 0.8
python3 -I $P/render.py $E build/timeline.json assets build --frames 0.0,<busy moments...>
python3 -I $P/sheet.py build/preview build/sheet.jpg 4 470     # LOOK: text clear of subject, nothing cut off, first frame readable
python3 -I $P/render.py $E build/timeline.json assets build
D=$(python3 -c "import json;print(json.load(open('build/timeline.json'))['duration'])")
python3 -I $P/music.py $D build/music.wav && python3 -I $P/mix.py build/narration.wav build/music.wav build/mix_raw.wav 19
cd build && ffmpeg -y -loglevel error -f concat -safe 0 -i concat.txt -c copy video_only.mp4 \
 && ffmpeg -y -loglevel error -i mix_raw.wav -af loudnorm=I=-14:TP=-1.5:LRA=11 -ar 48000 mix.wav \
 && ffmpeg -y -loglevel error -i video_only.mp4 -i mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -movflags +faststart -shortest ../out/<slug>.mp4
```
Check: transcribe the final file with faster-whisper and compare with the script; pull 8 frames with ffmpeg and look at them.

## 5. Host + schedule
- Copy the mp4 to `videos/`, commit, push. Public URL: `https://raw.githubusercontent.com/mrkyog/shorts-media/main/videos/<slug>.mp4` (check it returns 200).
- Metricool brand blogId `7999927`, timezone Asia/Taipei. `createScheduledPost` with providers tiktok, instagram, youtube:
  - `text`: 1–2 sentence caption + 3–5 hashtags.
  - `tiktokData.title` is REQUIRED (≤90 chars), `isAigc: true`.
  - `instagramData`: `{"type":"REEL","showReelOnFeed":true,"isAiGenerated":true}`.
  - `youtubeData`: `{"title":..., "type":"short","privacy":"public","category":"SCIENCE_TECHNOLOGY","madeForKids":false,"tags":[...]}`.
  - Never post to any other YouTube channel; only the one connected in this Metricool brand.
- If the target slot has already passed, schedule for now + 10 minutes.

## 6. Log
Append to `topics.md`: date, slot, slug, title, sources, Metricool post id. Commit and push.
