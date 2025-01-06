import threading
from pydub import AudioSegment
from pydub.playback import play
import  soundfile as sf
import sounddevice as sd



device_index = 27
# Функция для воспроизведения аудио
def play_audio_segment(file_path):
    audio = AudioSegment.from_mp3(file_path)
    play(audio)

def play_sounddevice(file_path, device_index):
    data, samplerate = sf.read(file_path)
    sd.default.device = device_index  # Указываем индекс устройства
    sd.play(data, samplerate)
    sd.wait()

def play_audio(file_path, device_index):
    thread1 = threading.Thread(target=play_audio_segment, args=(file_path,))
    thread2 = threading.Thread(target=play_sounddevice, args=(file_path, device_index))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
play_audio('base_answers/first_hello.wav',device_index)