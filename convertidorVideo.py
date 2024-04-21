from moviepy.editor import VideoFileClip

def convert_avi_to_mp4(avi_file_path, mp4_file_path):
    video = VideoFileClip(avi_file_path)
    video.write_videofile(mp4_file_path)

# Uso de la función
convert_avi_to_mp4('/home/nicolas/Videos/pruebas4.AVI', '/home/nicolas/Videos/pruebas4MP4.mp4')