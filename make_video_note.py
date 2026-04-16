from pathlib import Path
from moviepy import VideoFileClip

src = Path(r"C:\Users\user\Desktop\bot_server\assets\video_notes\yulia_test_01.mp4")
dst = Path(r"C:\Users\user\Desktop\bot_server\assets\video_notes\yulia_note_01.mp4")

clip = VideoFileClip(str(src))

size = int(min(clip.w, clip.h))

square = clip.cropped(
    x_center=clip.w / 2,
    y_center=clip.h / 2,
    width=size,
    height=size
)

final = square.resized(new_size=(512, 512))

final.write_videofile(
    str(dst),
    codec="libx264",
    audio_codec="aac",
    fps=clip.fps or 25
)

clip.close()
square.close()
final.close()

print("готово:", dst)