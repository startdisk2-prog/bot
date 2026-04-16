from pathlib import Path
from moviepy import VideoFileClip

BASE_DIR = Path(__file__).resolve().parent

INPUT_VIDEO = BASE_DIR / "assets" / "video_notes" / "yulia_test_01.mp4"
OUTPUT_VIDEO = BASE_DIR / "assets" / "video_notes" / "yulia_note_01.mp4"

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

# Настройки кадра
OUTPUT_SIZE = 640      # квадрат для Telegram video note
MAX_DURATION = 59      # безопасная длина
ZOOM = 1.18            # немного приблизить лицо
FOCUS_Y = 0.20         # сместить кадр вверх, чтобы не резало голову


def main():
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Не найден исходник: {INPUT_VIDEO}")

    clip = VideoFileClip(str(INPUT_VIDEO))

    if clip.duration > MAX_DURATION:
        clip = clip.subclipped(0, MAX_DURATION)

    w, h = clip.size

    # Делаем квадратный кроп без размытого фона
    crop_size = int(min(w, h) / ZOOM)
    crop_size = min(crop_size, w, h)

    max_x = max(0, w - crop_size)
    max_y = max(0, h - crop_size)

    x1 = int(max_x * 0.5)
    y1 = int(max_y * FOCUS_Y)

    cropped = clip.cropped(
        x1=x1,
        y1=y1,
        x2=x1 + crop_size,
        y2=y1 + crop_size
    )

    final = cropped.resized(new_size=(OUTPUT_SIZE, OUTPUT_SIZE))

    fps = int(getattr(clip, "fps", 25) or 25)

    final.write_videofile(
        str(OUTPUT_VIDEO),
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        preset="medium",
        bitrate="2500k"
    )

    final.close()
    cropped.close()
    clip.close()

    print(f"готово: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()