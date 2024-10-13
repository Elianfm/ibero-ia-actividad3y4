#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install requests beautifulsoup4 selenium==3.14.0 webdriver-manager


# In[5]:


import bs4
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import base64

# Creación de un directorio para guardar las imágenes
folder_name = 'imagenes/buñuelos'
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

def download_image(url, folder_name, num):
    # Descargando la imagen
    if url.startswith('http'):  # Verificar si la URL es válida
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(folder_name, str(num) + ".jpg"), 'wb') as file:
                file.write(response.content)
    elif url.startswith('data:image'):  # Verificar si la imagen está en formato base64
        try:
            header, encoded = url.split(',', 1)
            data = base64.b64decode(encoded)
            with open(os.path.join(folder_name, str(num) + ".jpg"), 'wb') as file:
                file.write(data)
        except Exception as e:
            print(f"Error al decodificar la imagen {num}: {e}")
    else:
        print(f"URL inválida para la imagen {num}: {url}")

# Inicialización del driver en Selenium 3.14
chromePath = ChromeDriverManager().install()  # Usando webdriver-manager para obtener el chromedriver
driver = webdriver.Chrome(executable_path=chromePath)

# URL de búsqueda en Google
search_URL = "https://www.google.com/search?q=buñuelos+colombianos&source=lnms&tbm=isch"
driver.get(search_URL)

# Esperar a que el usuario confirme antes de proceder
a = input("Waiting...")

# Hacer scroll hacia arriba
driver.execute_script("window.scrollTo(0, 0);")

# Obtener el HTML de la página y analizarlo con BeautifulSoup
soup = bs4.BeautifulSoup(driver.page_source, 'html.parser')

# Buscar todas las imágenes con la clase 'YQ4gaf' cuyo div padre tiene la clase 'H8Rx8c'
imgs = []
parent_divs = soup.find_all('div', class_='H8Rx8c')
for div in parent_divs:
    img = div.find('img', class_='YQ4gaf')
    if img:
        imgs.append(img)

# Descargar cada imagen encontrada
for idx, img in enumerate(imgs):
    img_url = img.get('src')
    if img_url:
        download_image(img_url, folder_name, idx)

# Imprimir la cantidad de imágenes descargadas
print(f"Se descargaron {len(imgs)} imágenes con la clase 'YQ4gaf' dentro de un div con la clase 'H8Rx8c'.")


# In[ ]:


print("termino??")


# In[ ]:




