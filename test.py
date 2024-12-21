import g4f
import json
import vosk
import pyaudio
import torch
import sounddevice as sd
import numpy as np
import keyboard  # Библиотека для обработки нажатий клавиш

def voice_Stella():
    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'v4_ru.pt'
    speaker = 'kseniya'  # 'aidar', 'baya', 'kseniya', 'xenia', 'random'
    sample_rate = 24000  # 8000, 24000, 48000
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    # Разбиваем текст на части по 1000 символов
    chunks = [a[i:i + 1000] for i in range(0, len(a), 1000)]
    audio = np.array([])

    for chunk in chunks:
        audio_chunk = model.apply_tts(text=chunk, speaker=speaker, sample_rate=sample_rate)
        audio = np.concatenate((audio, audio_chunk.numpy()))

    # Воспроизведение аудио
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Ожидание завершения воспроизведения

def vosk_rec():
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
            if text != '':
                print(text)
                print("Нажмите Enter для подтверждения, Backspace для повторного распознавания или Shift для ручного ввода.")
                while True:
                    if keyboard.is_pressed('enter'):
                        print("Отправлено")
                        return text
                    elif keyboard.is_pressed('backspace'):
                        print("Говорите...")
                        break
                    elif keyboard.is_pressed('shift'):
                        manual_input = input("Введите текст вручную: ")
                        print("Отправлено")
                        return manual_input

a = ''

def ask_gpt(messages) -> str:
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4
        ,
        messages=messages)
    global a
    a = response
    return response

start_prompt = "role-playing game 16 yars now girl-boyfriend relationship: You're my girlfriend, and I'm your boyfriend, you talk to me, you're funny, you like to flirt. Answer as much as possible as a person, answer only in Russian. Now start the conversation with the phrase: hi, nice guy"

messages = []
messages.append({"role": "user", "content": start_prompt})
messages.append({"role": "assistant", "content": ask_gpt(messages)})
voice_Stella()

while True:
    messages.append({"role": "user", "content": vosk_rec()})
    messages.append({"role": "assistant", "content": ask_gpt(messages)})
    voice_Stella()
