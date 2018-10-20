import time
import pyocr
from PIL import Image
import pyocr.builders

def get_digit_ocr_info(img):
    result = None
    start_time = time.time()
    print("*****start convert_image_to_deadline *****")

    width, height = img.size

    tools = pyocr.get_available_tools()
    tool = tools[0]
    print(tool)
    langs = tool.get_available_languages()
    print("support langs: %s"%",".join(langs))

    lang = 'eng'

    digit_text = tool.image_to_string(
        img,
        lang=lang,
        builder=pyocr.builders.DigitBuilder(tesseract_layout=6)
    )
    print('DigitBuilder', digit_text)

    print('******** end convert_image_to_deadline *******')

    return digit_text