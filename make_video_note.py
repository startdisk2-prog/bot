from pathlib import Path
from moviepy import VideoFileClip, ColorClip, CompositeVideoClip

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "assets" / "video_notes"

# СЮДА должен лежать ИСХОДНИК из HeyGen, не готовый кружок
SOURCE_VIDEO = VIDEO_DIR / "avatar_source.mp4"

# Это уже готовый результат
OUTPUT_VIDEO = VIDEO_DIR / "yulia_note_01.mp4"

FINAL_SIZE = 512
INNER_HEIGHT = 430   # меньше = голова дальше, в кружке помещается лучше
BG_COLOR = (18, 18, 18)


def main():
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(
            f"Не найден файл:\n{SOURCE_VIDEO}\n\n"
            f"Положи сюда исходное видео из HeyGen и назови его avatar_source.mp4"
        )

    if SOURCE_VIDEO.resolve() == OUTPUT_VIDEO.resolve():
        raise ValueError("SOURCE_VIDEO и OUTPUT_VIDEO не должны быть одним и тем же файлом")

    clip = VideoFileClip(str(SOURCE_VIDEO))

    # Вписываем видео внутрь квадрата БЕЗ двойной подложки
    fitted = clip.resized(height=INNER_HEIGHT)

    # Если вдруг после resize ширина слишком большая — дополнительно ограничим
    if fitted.w > FINAL_SIZE - 20:
        fitted = fitted.resized(width=FINAL_SIZE - 20)

    bg = ColorClip(size=(FINAL_SIZE, FINAL_SIZE), color=BG_COLOR, duration=clip.duration)

    final = CompositeVideoClip(
        [
            bg,
            fitted.with_position(("center", "center"))
        ],
        size=(FINAL_SIZE, FINAL_SIZE)
    )

    if clip.audio is not None:
        final = final.with_audio(clip.audio)

    final.write_videofile(
        str(OUTPUT_VIDEO),
        codec="libx264",
        audio_codec="aac",
        fps=clip.fps or 25,
        preset="medium"
    )

    clip.close()
    fitted.close()
    bg.close()
    final.close()

    print(f"готово: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()