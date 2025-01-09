import asyncio
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Timer
import g4f
import pyaudio
import torch
import vosk
from torch.cuda.amp import autocast
from skills import *
import words
import threading
from pydub import AudioSegment
from pydub.playback import play
import soundfile as sf
import sounddevice as sd

device_index = 11

# Функция инициализации
def initialize():
    global vectorizer, clf, model, recognizer, p, stream

    # Загрузка моделей и векторизатора
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Загрузка модели Vosk
    model = vosk.Model("vosk_model")
    recognizer = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()

# Вызов функции инициализации перед основным процессом
initialize()

text_to_tts = ''
recognizer_active = True
trigger_active = False

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
    func_name = answer.split()[0]

    global text_to_tts
    text_to_tts = answer.replace(func_name, '')
    asyncio.run(voice_Stella())

    exec(func_name + '()')

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

def ask_gpt(messages) -> str:
    global text_to_tts
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=messages
    )
    text_to_tts = response
    return response

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

def chatting_mode():
    start_prompt = "role-playing game 16 yars now girl-boyfriend relationship: You're my girlfriend, and I'm your boyfriend, you talk to me, you're funny, you like to flirt. Answer as much as possible as a person, answer only in Russian. Now start the conversation with the phrase: hi, nice guy"

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

play_audio('base_answers/first_hello.wav', device_index)

vosk_rec()
