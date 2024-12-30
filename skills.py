# skills.py
import webbrowser
import sys
import os
from pydub import AudioSegment
from pydub.playback import play
import config  # Импортируем config

# Функция для воспроизведения аудио
def play_audio(file_path):
    audio = AudioSegment.from_mp3(file_path)
    play(audio)

# Функция для запуска браузера
def browser():
    '''Открывает браузер, заданный по умолчанию в системе с URL, указанным здесь'''
    play_audio('base_answers/launching_browser.mp3')
    webbrowser.open('https://www.youtube.com', new=2)

# Функция для выключения компьютера
def offpc():
    play_audio('base_answers/shutting_down_computer.mp3')
    os.system('shutdown \s')

# Функция для отключения бота
def offBot():
    '''Отключает бота'''
    play_audio('base_answers/disconnecting.mp3')
    sys.exit()

# Функция для запуска Яндекс.Музыки
def run_yandexmusic():
    play_audio('base_answers/playing_music.mp3')
    os.startfile('yndmusic.exe.lnk')

# Функция для запуска командной строки
def run_cmd():
    play_audio('base_answers/launching_cmd.mp3')
    os.startfile('cmd.exe.lnk')

# Функция для запуска Discord
def run_discord():
    play_audio('base_answers/launching_discord.mp3')
    os.startfile('Discord_run.lnk')

# Функция для поиска в браузере
def search_browser():
    parts = config.vosk_rec_text.split('найди', 1)
    if len(parts) > 1:
        query_txt = parts[1]
        play_audio('base_answers/opening_browser.mp3')
        webbrowser.open(url=f'https://www.google.com/search?q={query_txt}')

# Функция для показа погоды
def weather():
    play_audio('base_answers/opening.mp3')
    webbrowser.open(url='https://yandex.ru/pogoda/volgograd')

# Функция для работы с интернетом
def internet_activated():
    webbrowser.open('https://www.youtube.com', new=2)
    play_audio('base_answers/internet_activated.mp3')

# Функция для приветствия
def hello_master():
    play_audio('base_answers/hello_master.mp3')

# Функция для очищения окон
def cleaned_windows():
    play_audio('base_answers/cleaned_windows.mp3')

# Функция для ответа "Как дела"
def as_you_see():
    play_audio('base_answers/as_you_see.mp3')

# Функция для запуска командной строки
def cmd_activated():
    os.startfile('cmd.exe.lnk')
    play_audio('base_answers/cmd_activated.mp3')

# Функция для активации Discord
def discord_activated():
    os.startfile('Discord_run.lnk')
    play_audio('base_answers/discord_activated.mp3')

# Функция для начала работы
def opening():
    play_audio('base_answers/opening.mp3')
def working_in_background():
    play_audio('base_answers/working_in_background.mp3')
def opening_browser_now():
    play_audio('base_answers/opening_browser_now.mp3')
def seems_like_it():
    play_audio('base_answers/seems_like_it.mp3')#
def i_can_do_many_things():
    play_audio('base_answers/i_can_do_many_things.mp3')