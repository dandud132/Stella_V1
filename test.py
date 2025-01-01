import asyncio
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import g4f
import pyaudio
import torch
import vosk

import config
from skills import *
import words

# Загрузка моделей и векторизатора
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    clf = pickle.load(f)

text_to_tts = ''
recognizer_active = True

def recognize(data, vectorizer, clf):
    trg = words.TRIGGERS.intersection(data.split())
    if not trg:
        return
    print('распознано')

    data = data.replace(list(trg)[0], '')
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

    device = torch.device('cuda')
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
vosk_rec()