# skills.py
import threading
import webbrowser
import sys
import os
from os import startfile
from pydub import AudioSegment
from pydub.playback import play
import config  # Импортируем config
import  soundfile as sf
import sounddevice as sd



device_index = 2
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

# Функция для запуска браузера
def browser():
    '''Открывает браузер, заданный по умолчанию в системе с URL, указанным здесь'''
    play_audio('base_answers/launching_browser.mp3', device_index)
    webbrowser.open('https://www.youtube.com', new=2)

# Функция для выключения компьютера
def offpc():
    play_audio('base_answers/shutting_down_computer.mp3', device_index)
    os.system('shutdown /s')

# Функция для отключения бота
def offBot():
    '''Отключает бота'''
    play_audio('base_answers/disconnecting.mp3', device_index)
    sys.exit()

# Функция для запуска Яндекс.Музыки
def run_yandexmusic():
    play_audio('base_answers/playing_music.mp3', device_index)
    os.startfile('yndmusic.exe.lnk')

# Функция для запуска командной строки
def run_cmd():
    play_audio('base_answers/launching_cmd.mp3', device_index)
    os.startfile('cmd.exe.lnk')

# Функция для запуска Discord
def run_discord():
    play_audio('base_answers/launching_discord.mp3', device_index)
    os.startfile('Discord_run.lnk')

# Функция для поиска в браузере
def search_browser():
    parts = config.vosk_rec_text.split('найди', 1)
    if len(parts) > 1:
        query_txt = parts[1]
        play_audio('base_answers/opening_browser.mp3', device_index)
        webbrowser.open(url=f'https://www.google.com/search?q={query_txt}')

# Функция для показа погоды
def weather():
    play_audio('base_answers/opening.mp3', device_index)
    webbrowser.open(url='https://yandex.ru/pogoda/volgograd')

# Функция для работы с интернетом
def internet_activated():
    webbrowser.open('https://www.youtube.com', new=2)
    play_audio('base_answers/internet_activated.mp3', device_index)

# Функция для приветствия
def hello_master():
    play_audio('base_answers/hello_master.mp3', device_index)

# Функция для очищения окон
def cleaned_windows():
    play_audio('base_answers/cleaned_windows.mp3', device_index)

# Функция для ответа "Как дела"
def as_you_see():
    play_audio('base_answers/as_you_see.mp3', device_index)

# Функция для запуска командной строки
def cmd_activated():
    os.startfile('cmd.exe.lnk')
    play_audio('base_answers/cmd_activated.mp3', device_index)

# Функция для активации Discord
def discord_activated():
    os.startfile('Discord_run.lnk')
    play_audio('base_answers/discord_activated.mp3', device_index)

# Функция для начала работы
def opening():
    play_audio('base_answers/opening.mp3', device_index)
def working_in_background():
    play_audio('base_answers/working_in_background.mp3', device_index)
def opening_browser_now():
    play_audio('base_answers/opening_browser_now.mp3', device_index)
def seems_like_it():
    play_audio('base_answers/seems_like_it.mp3', device_index)  #
def i_can_do_many_things():
    play_audio('base_answers/i_can_do_many_things.mp3', device_index)

# Функция для запуска YouTube и поиска видео
def search_youtube():
    print(config.vosk_rec_text)
    query = config.vosk_rec_text.split('видео', 1)
    query = [q.strip() for q in query if q.strip()]  # Удаление пустых строк
    query = ' '.join(query)  # Объединение текста в одну строку
    play_audio('base_answers/search_youtube.mp3', device_index)
    webbrowser.open(f'https://www.youtube.com/results?search_query={query}', new=2)
def waiting_for_next_command():
    play_audio('base_answers/waiting_for_next_command.mp3', device_index)

def start_play_pause():
    os.startfile('k.ahk')


def next_track():
    play_audio('base_answers/next_track.mp3', device_index)
    os,startfile('n.ahk')
def past_track():
    play_audio('base_answers/past_track.mp3', device_index)
    os.startfile('p.ahk')
def pause():
    play_audio('base_answers/pause.mp3', device_index)
    os.startfile('k.ahk')
def play_m(device_index=27):
    play_audio('base_answers/play.mp3', device_index)
    os.startfile('k.ahk')

