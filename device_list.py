import sounddevice as sd

# Получение списка аудиоустройств
devices = sd.query_devices()

# Вывод списка аудиоустройств
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")
