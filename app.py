import g4f
import json
import vosk
import pyaudio
import torch
import sounddevice as sd
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor
import words

with open('vectorizer.pkl', 'rb') as f: vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f: clf = pickle.load(f)
text_to_tts = ''
recognizer_active = True

def recognize(data, vectorizer, clf):
    '''
    Анализ распознанной речи
    '''

    #проверяем есть ли имя бота в data, если нет, то return
    trg = words.TRIGGERS.intersection(data.split())
    if not trg:
        print('7')
        return

    #удаляем имя бота из текста
    data.replace(list(trg)[0], '')

    #получаем вектор полученного текста
    #сравниваем с вариантами, получая наиболее подходящий ответ
    text_vector = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vector])[0]

    #получение имени функции из ответа из data_set
    func_name = answer.split()[0]

    #озвучка ответа из модели data_set
    global text_to_tts
    text_to_tts = answer.replace(func_name, '')
    voice_Stella()


    #запуск функции из skills
    exec(func_name + '()')

def voice_Stella():
    global recognizer_active
    recognizer_active = False  # Останавливаем распознавание речи

    device = torch.device('cuda')
    torch.set_num_threads(4)
    local_file = 'v4_ru.pt'
    speaker = 'kseniya'  # 'aidar', 'baya', 'kseniya', 'xenia', 'random'
    sample_rate = 24000  # 8000, 24000, 48000
    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    # Разбиваем текст на части по 1000 символов
    chunks = [text_to_tts[i:i + 1100] for i in range(0, len(text_to_tts), 1000)]
    audio = np.array([])

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(model.apply_tts, text=chunk, speaker=speaker, sample_rate=sample_rate) for chunk in chunks]
        for future in futures:
            audio_chunk = future.result()
            audio = np.concatenate((audio, audio_chunk.numpy()))

    # Воспроизведение аудио
    sd.play(audio, samplerate=sample_rate)
    sd.wait()  # Ожидание завершения воспроизведения

    recognizer_active = True  # Возобновляем распознавание речи

def ask_gpt(messages) -> str:
    global text_to_tts
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4,
        messages=messages
    )
    text_to_tts = response
    return response

def chatting_mode():
    start_prompt = "role-playing game 16 yars now girl-boyfriend relationship: You're my girlfriend, and I'm your boyfriend, you talk to me, you're funny, you like to flirt. Answer as much as possible as a person, answer only in Russian. Now start the conversation with the phrase: hi, nice guy"

    messages = [{"role": "user", "content": start_prompt}]
    messages.append({"role": "assistant", "content": ask_gpt(messages)})
    voice_Stella()

    while True:
        user_input = chatting_rec()
        if user_input:
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": ask_gpt(messages)})
            voice_Stella()
        else:
            break

    vosk_rec()

def vosk_rec():
    global recognizer_active
    model = vosk.Model("vosk_model")
    recognizer = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer_active and recognizer.AcceptWaveform(data):
            answer = recognizer.Result()
            text = json.loads(answer)["text"]
            if text:
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
                print(text)
                return text
            else:
                print(text)
                return None

vosk_rec()
