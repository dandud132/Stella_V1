import customtkinter as ctk
import threading
import asyncio
import json
import pickle
import pyaudio
import vosk
import torch
import g4f
import config  # Импортируем config
from threading import Timer
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import sounddevice as sd
from torch.cuda.amp import autocast
from skills import *  # Импортируем все функции из skills.py
import words
import sys

# Параметры устройства
device_index = 11

# Инициализация ассистента
def initialize():
    global vectorizer, clf, model, recognizer, p, stream

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    model = vosk.Model("vosk_model")
    recognizer = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()

initialize()

# Статус и триггер
text_to_tts = ''
recognizer_active = True
trigger_active = False

# Функции для проигрывания звука
def play_audio_segment(file_path):
    audio = AudioSegment.from_mp3(file_path)
    play(audio)

def play_sounddevice(file_path, device_index):
    data, samplerate = sf.read(file_path)
    sd.default.device = device_index
    sd.play(data, samplerate)
    sd.wait()

def play_audio(file_path, device_index):
    thread1 = threading.Thread(target=play_audio_segment, args=(file_path,))
    thread2 = threading.Thread(target=play_sounddevice, args=(file_path, device_index))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

def reset_trigger():
    global trigger_active
    trigger_active = False

def recognize(data, vectorizer, clf):
    global trigger_active
    trg = words.TRIGGERS.intersection(data.split())
    if not trg and not trigger_active:
        return
    print('распознано')

    if trg:
        data = data.replace(list(trg)[0], '')
        trigger_active = True
        Timer(20, reset_trigger).start()

    text_vector = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vector])[0]
    func_name = answer.split()[0]  # Получаем название функции

    global text_to_tts
    text_to_tts = answer.replace(func_name, '')

    # Используем словарь функций для вызова
    if func_name in function_map:
        function_map[func_name]()  # Вызываем нужную функцию из словаря
    else:
        print(f"Функция '{func_name}' не найдена.")

# Словарь для хранения всех функций из skills.py
function_map = {
    'i_can_do_many_things': i_can_do_many_things,
    'browser': browser,
    'offpc': offpc,
    'offBot': offBot,
    'run_yandexmusic': run_yandexmusic,
    'run_cmd': run_cmd,
    'run_discord': run_discord,
    'search_browser': search_browser,
    'weather': weather,
    'internet_activated': internet_activated,
    'hello_master': hello_master,
    'cleaned_windows': cleaned_windows,
    'as_you_see': as_you_see,
    'cmd_activated': cmd_activated,
    'discord_activated': discord_activated,
    'opening': opening,
    'working_in_background': working_in_background,
    'opening_browser_now': opening_browser_now,
    'seems_like_it': seems_like_it,
    'search_youtube': search_youtube,
    'waiting_for_next_command': waiting_for_next_command,
    'start_play_pause': start_play_pause,
    'next_track': next_track,
    'past_track': past_track,
    'pause': pause,
    'play_m': play_m,
}

# Асинхронная функция для синтеза речи
async def voice_Stella():
    global recognizer_active
    recognizer_active = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    local_file = 'v4_ru.pt'
    speaker = 'kseniya'
    sample_rate = 24000
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    chunks = [text_to_tts[i:i + 1000] for i in range(0, len(text_to_tts), 1000)]
    audio = AudioSegment.silent(duration=0)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(model.apply_tts, text=chunk, speaker=speaker, sample_rate=sample_rate) for chunk in chunks]
        for future in futures:
            with autocast():
                audio_chunk = future.result().numpy()
                audio_chunk = AudioSegment(
                    audio_chunk.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=audio_chunk.dtype.itemsize,
                    channels=1
                )
                audio += audio_chunk + AudioSegment.silent(duration=100)

    play(audio)
    recognizer_active = True

# Функция для получения ответа от GPT
def ask_gpt(messages) -> str:
    global text_to_tts
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=messages
    )
    text_to_tts = response
    return response

# Функция для распознавания речи с использованием Vosk
def vosk_rec():
    global recognizer_active
    with ThreadPoolExecutor() as executor:
        print('говорите')
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer_active and recognizer.AcceptWaveform(data):
                future = executor.submit(recognizer.Result)
                answer = future.result()
                text = json.loads(answer)["text"]
                if text:
                    if text == 'давай поболтаем':
                        chatting_mode()
                    else:
                        config.vosk_rec_text = text
                        print(text)
                        recognize(data=text, vectorizer=vectorizer, clf=clf)

# Функция для начала общения с GPT (в режиме чат-игры)
def chatting_rec():
    model = vosk.Model("vosk_model")
    recognizer = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            answer = recognizer.Result()
            text = json.loads(answer)["text"]
            if text != 'пока':
                return text
            else:
                return None

# Режим общения с GPT
def chatting_mode():
    start_prompt = "role-playing game 16 years now girl-boyfriend relationship: You're my girlfriend, and I'm your boyfriend, you talk to me, you're funny, you like to flirt. Answer as much as possible as a person, answer only in Russian. Now start the conversation with the phrase: hi, nice guy"

    messages = [{"role": "user", "content": start_prompt}]
    messages.append({"role": "assistant", "content": ask_gpt(messages)})
    asyncio.run(voice_Stella())

    while True:
        user_input = chatting_rec()
        if user_input:
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": ask_gpt(messages)})
            asyncio.run(voice_Stella())
        else:
            break

    vosk_rec()

# customtkinter GUI для запуска/остановки ассистента с меню смены темы в настройках
class AssistantGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Голосовой Ассистент")
        self.geometry("400x400")

        # Настроим стиль
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Создадим Tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        # Вкладки
        self.tabview.add("Главная")
        self.tabview.add("Настройки")

        # Вкладка "Главная"
        self.main_tab = self.tabview.tab("Главная")
        self.status_label = ctk.CTkLabel(self.main_tab, text="Ассистент выключен", font=("Arial", 16))
        self.status_label.pack(pady=20)

        self.start_button = ctk.CTkButton(self.main_tab, text="Запустить Ассистента", command=self.start_assistant, font=("Arial", 14), fg_color="#4682B4", text_color="black")
        self.start_button.pack(pady=10)

        self.stop_button = ctk.CTkButton(self.main_tab, text="Остановить Ассистента", command=self.stop_assistant, font=("Arial", 14), fg_color="#4682B4", text_color="black")
        self.stop_button.pack(pady=10)
        self.stop_button.configure(state="disabled")

        # Вкладка "Настройки"
        self.settings_tab = self.tabview.tab("Настройки")
        self.theme_label = ctk.CTkLabel(self.settings_tab, text="Выберите тему:", font=("Arial", 12))
        self.theme_label.pack(pady=10)

        self.theme_menu = ctk.CTkOptionMenu(self.settings_tab, values=["light", "dark"], command=self.change_theme, fg_color="#4682B4")
        self.theme_menu.pack(pady=10)

    def start_assistant(self):
        play_audio('base_answers/first_hello.wav', device_index)  # Воспроизведение приветствия
        self.status_label.configure(text="Ассистент включен")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        assistant_thread = threading.Thread(target=self.run_assistant)
        assistant_thread.start()

    def stop_assistant(self):
        global recognizer_active
        recognizer_active = False
        self.status_label.configure(text="Ассистент выключен")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.quit()
        sys.exit()

    def change_theme(self, theme):
        ctk.set_appearance_mode(theme)

    def run_assistant(self):
        vosk_rec()

# Запуск GUI
if __name__ == "__main__":
    app = AssistantGUI()
    app.mainloop()
