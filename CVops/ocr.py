from langdetect import detect_langs
from PIL import Image
import pytesseract

def detect_lang_with_langdetect(text):
    try:
        langs = detect_langs(text)
        for item in langs:
            return item.lang, item.prob
    except:
        return "err", 0.0


#function to process image and detect text

def detect_text_in_image(image_path):
    with Image.open(image_path) as img:
        text = pytesseract.image_to_string(img, lang = 'deu')

        return text

image_path = '/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/01Schraubedrawcorrected/page_1.jpg'

text = detect_text_in_image(image_path)

#detect language of extarcted text
lang_detect_result = detect_language_with_langdetect(text)
print("Langdetect detected language:{Lang_detect_result[0]} with probability: {Lang_detect_result[1]} ")

if Lang_detect_result [0] == 'de':
    print('text is in german')
else:
    print("not in german")