from pathlib import Path
from moviepy import VideoFileClip, CompositeVideoClip

BASE_DIR = Path(__file__).resolve().parent

SOURCE_VIDEO = BASE_DIR / "assets" / "video_notes" / "yulia_test_01.mp4"
OUTPUT_VIDEO = BASE_DIR / "assets" / "video_notes" / "yulia_note_01.mp4"

SIZE = 512
FOREGROUND_HEIGHT = 470   # меньше число = дальше лицо
FOREGROUND_Y_SHIFT = 18   # больше число = чуть ниже в кружке
FPS = 30


def make_cover_square(clip, size: int):
    if clip.w >= clip.h:
        base = clip.resized(height=size)
    else:
        base = clip.resized(width=size)

    return base.cropped(
        x_center=base.w / 2,
        y_center=base.h / 2,
        width=size,
        height=size,
    )


def build_video_note():
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"Не найден исходник: {SOURCE_VIDEO}")

    source = VideoFileClip(str(SOURCE_VIDEO))

    background = make_cover_square(source, SIZE)

    foreground = source.resized(height=FOREGROUND_HEIGHT)
    fg_x = (SIZE - foreground.w) / 2
    fg_y = ((SIZE - foreground.h) / 2) + FOREGROUND_Y_SHIFT

    foreground = foreground.with_position((fg_x, fg_y))

    final = CompositeVideoClip(
        [background, foreground],
        size=(SIZE, SIZE)
    ).with_duration(source.duration).with_audio(source.audio)

    final.write_videofile(
        str(OUTPUT_VIDEO),
        codec="libx264",
        audio_codec="aac",
        fps=FPS
    )

    source.close()
    background.close()
    foreground.close()
    final.close()

    print(f"готово: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    build_video_note()