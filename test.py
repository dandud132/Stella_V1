import g4f

def voice_Stella():
    import torch
    import sounddevice as sd  # Добавлено для воспроизведения аудио
    import numpy as np  # Добавлено для обработки аудио

    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'v4_ru.pt'
    speaker = 'kseniya'  # 'aidar', 'baya', 'kseniya', 'xenia', 'random'
    sample_rate = 24000  # 8000, 24000, 48000

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    example_text = a
    audio = model.apply_tts(text=example_text,
                            speaker=speaker,
                            sample_rate=sample_rate)

    # Воспроизведение аудио
    audio = audio.numpy()  # Преобразование тензора в numpy массив
    sd.play(audio, samplerate=sample_rate)  # Воспроизведение аудио
    sd.wait()  # Ожидание завершения воспроизведения
def vosk_rec():
    import json
    import vosk
    import pyaudio

    # Инициализация микрофона
    model = vosk.Model("vosk_model")
    recognizer = vosk.KaldiRecognizer(model, 16000)

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)
    stream.start_stream()

    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            answer = recognizer.Result()
            text = json.loads(answer)["text"]  # Преобразование результата в строку
            if text != '':
                print(text)
                return text


a = ''
def ask_gpt(messages) -> str:
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_35_turbo,
        messages = messages)

    global a
    a = response

messages = []
messages.append({"role": "user", "content": 'ты виртуальная девушка Стелла'})
messages.append({"role": "assistants", "content": ask_gpt(messages)})
voice_Stella()
while True:
    messages.append({"role": "user", "content": vosk_rec()})
    messages.append({"role": "assistants", "content": ask_gpt(messages)})
    voice_Stella()


