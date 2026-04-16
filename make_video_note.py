from pathlib import Path
from moviepy import VideoFileClip, CompositeVideoClip, ColorClip

ROOT_DIR = Path(__file__).resolve().parent

SOURCE_CANDIDATES = [
    ROOT_DIR / "assets" / "video_notes" / "yulia_test_01.mp4",
    ROOT_DIR / "assets" / "refs" / "yulia_video.mov",
]

OUTPUT_VIDEO = ROOT_DIR / "assets" / "video_notes" / "yulia_note_01.mp4"

SIZE = 640
FPS_FALLBACK = 30

# Чем меньше значение, тем "дальше" Юля от камеры и тем меньше риск,
# что голова будет срезана.
FOREGROUND_HEIGHT_RATIO = 0.80

# Чем больше значение, тем ниже Юля будет стоять в кадре.
FOREGROUND_TOP_RATIO = 0.15

# Ограничение ширины переднего слоя, чтобы кружок не выглядел слишком тесным.
FOREGROUND_MAX_WIDTH_RATIO = 0.82

# Немного затемняем фон, чтобы главный слой выглядел аккуратнее.
BACKGROUND_DARKNESS = 0.22


def pick_source_video() -> Path:
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Не найден исходный файл. "
        "Положи исходное видео в assets/video_notes/yulia_test_01.mp4 "
        "или в assets/refs/yulia_video.mov"
    )


def make_square_background(clip: VideoFileClip, size: int):
    # Заполняем квадрат фоном из того же видео
    if clip.w < clip.h:
        bg = clip.resized(width=size)
    else:
        bg = clip.resized(height=size)

    x1 = max(0, (bg.w - size) / 2)
    y1 = max(0, (bg.h - size) / 2)

    bg = bg.cropped(
        x1=x1,
        y1=y1,
        x2=x1 + size,
        y2=y1 + size
    )
    return bg


def build_video_note():
    source_video = pick_source_video()
    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    clip = VideoFileClip(str(source_video))

    # Фон
    background = make_square_background(clip, SIZE)

    dark_overlay = ColorClip(
        size=(SIZE, SIZE),
        color=(0, 0, 0),
        duration=clip.duration
    ).with_opacity(BACKGROUND_DARKNESS)

    # Главный слой — делаем Юлю чуть меньше и чуть ниже
    foreground = clip.resized(height=int(SIZE * FOREGROUND_HEIGHT_RATIO))

    max_width = int(SIZE * FOREGROUND_MAX_WIDTH_RATIO)
    if foreground.w > max_width:
        foreground = foreground.resized(width=max_width)

    foreground_y = int(SIZE * FOREGROUND_TOP_RATIO)
    foreground = foreground.with_position(("center", foreground_y))

    final = CompositeVideoClip(
        [background, dark_overlay, foreground],
        size=(SIZE, SIZE)
    ).with_duration(clip.duration)

    if clip.audio is not None:
        final = final.with_audio(clip.audio)

    fps = getattr(clip, "fps", None) or FPS_FALLBACK

    final.write_videofile(
        str(OUTPUT_VIDEO),
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        preset="medium",
        bitrate="2500k",
        ffmpeg_params=["-movflags", "+faststart"]
    )

    try:
        clip.close()
    except Exception:
        pass

    try:
        background.close()
    except Exception:
        pass

    try:
        foreground.close()
    except Exception:
        pass

    try:
        final.close()
    except Exception:
        pass

    print(f"готово: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    build_video_note()