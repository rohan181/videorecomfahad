from flask import Flask, request, jsonify
import youtube_dl

app = Flask(__name__)

@app.route('/download', methods=['POST'])
def download_mp3():
    try:
        url = request.json['url']
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'mp3',
            'outtmpl': 'downloads/%(title)s.%(ext)s',
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            downloaded_filename = ydl.prepare_filename(info_dict)

        return jsonify({"message": "Downloaded successfully", "filename": downloaded_filename})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
