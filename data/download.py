# !pip install yt-dlp

from yt_dlp import YoutubeDL

def download_video(url, output_path):
    youtube_dl_options = {
        'format_sort': ['res:1080', 'ext:mp4:m4a'],
        "outtmpl": output_path,
    }
    with YoutubeDL(youtube_dl_options) as ydl:
        return ydl.download([url])

download_video("https://www.youtube.com/watch?v=72Qsjgi6j24", 'news1.mp4')
download_video("https://www.youtube.com/watch?v=LUAz1ffforY", 'news2.mp4')
download_video("https://www.youtube.com/watch?v=CL13X-8o4h0", 'mv1.mp4')
download_video("https://www.youtube.com/watch?v=CgCVZdcKcqY", 'mv2.mp4')
download_video("https://www.youtube.com/watch?v=LTEVRSdcn6U", 'lifelog1.mp4')
download_video("https://www.youtube.com/watch?v=lcQAT4DZsYk", 'lifelog2.mp4')
download_video("https://www.youtube.com/watch?v=8-m0lVBgUpo", 'ads1.mp4')
download_video("https://www.youtube.com/watch?v=0gVswArg7L8", 'ads2.mp4')