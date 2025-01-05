import soundfile as sf
import torch

# Загрузка модели
model_path = 'v4_ru.pt'
model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")

# Проверка, что модель загружена корректно
if model is None:
    raise ValueError("Не удалось загрузить модель")


# Функция для генерации аудио из текста
def generate_audio(text, output_file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Генерация аудио
    audio = model.apply_tts(text=text, speaker='kseniya', sample_rate=24000)

    # Сохранение аудио в файл
    sf.write(output_file, audio.cpu().numpy(), 24000)


# Пример использования функции
text = "Сейчас найду на ютуб"
output_file = "search_youtube.wav"
generate_audio(text, output_file)
