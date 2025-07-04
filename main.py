import subprocess

def extract_with_opensmile(audio_path, output_csv):
    cmd = [
        "./opensmile/build/progsrc/smilextract/Release/SMILExtract.exe",
        "-C", "opensmile/config/gemaps/v01a/GeMAPSv01a.conf",
        "-I", audio_path,
        "-O", output_csv
    ]
    subprocess.run(cmd)

extract_with_opensmile("WhatsApp Audio 2025-07-03 at 17.11.51_77c6b56d.wav", "features.csv")
