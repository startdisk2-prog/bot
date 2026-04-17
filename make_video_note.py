from pathlib import Path
from moviepy import VideoFileClip

BASE_DIR = Path(__file__).resolve().parent
SOURCE = BASE_DIR / "assets" / "video_notes" / "avatar_source.mp4"
OUTPUT = BASE_DIR / "assets" / "video_notes" / "yulia_note_01.mp4"

if not SOURCE.exists():
    raise FileNotFoundError(f"Не найден файл: {SOURCE}")

clip = VideoFileClip(str(SOURCE))
w, h = clip.size

# Делаем честный квадрат без дублей и без фоновых вставок
side = min(w, h)

# По горизонтали — центр
x1 = (w - side) // 2

# По вертикали — берем верхнюю часть, чтобы голова точно вошла в кружок
y1 = 0

x2 = x1 + side
y2 = y1 + side

square = clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2).resized((512, 512))

square.write_videofile(
    str(OUTPUT),
    codec="libx264",
    audio_codec="aac",
    fps=clip.fps
)

clip.close()
square.close()

print(f"готово: {OUTPUT}")