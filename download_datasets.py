import requests
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup

keywords = ["dogs","cats"]  # список ключевых слов для поиска изображений
num_images = 10000  # количество изображений для загрузки

for keyword in keywords:
    url = f"https://www.google.com/search?q={keyword}&tbm=isch"

    # создание папки для сохранения изображений
    folder_name = keyword.replace(" ", "_")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print(f"Поиск изображений для '{keyword}'")

    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    count = 0

    
    for i in tqdm(range(0, num_images, 100)):
        params = {"ijn": i}
        r = requests.get(url, params=params)
        soup = BeautifulSoup(r.text, 'html.parser')

        # поиск всех изображений на странице
        images = soup.find_all("img", {"class": "rg_i"})

        for image in images:
            link = image["src"]
            try:
                r = requests.get(link, stream=True)
                r.raw.decode_content = True

                with open(f"{folder_name}/{count}.jpg", "wb") as f:
                    shutil.copyfileobj(r.raw, f)

                count += 1
                if count == num_images:
                    break
            except:
                print("Не удалось загрузить изображение")

    print(f"Загружено {count} изображений '{keyword}'")

print("Готово! Всего загружено изображений:", count)