from PIL import Image, ImageOps
from pathlib import Path
import os

import cv2
import numpy as np
import svgwrite
import rembg
import aspose.words as aw


from PIL import Image
import cv2
import numpy as np
from PIL import Image


import cv2
import numpy as np
from PIL import Image

import sys
from PIL import Image
from potrace import Bitmap, POTRACE_TURNPOLICY_MINORITY  # `potracer` library



def file_to_svg(filename: str):
    try:

        image = Image.open(filename)
        width, height = image.size
        scale_factor = 1
        color_count = 2
        image = image.resize((width // scale_factor, height // scale_factor))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.quantize(colors=color_count).convert("RGB")

        grayscale_img = image.convert('L')






        gray_array = np.array(grayscale_img)

        gray_array = cv2.bilateralFilter(gray_array, d=9, sigmaColor=100, sigmaSpace=75)

        edges = cv2.Canny(gray_array, threshold1=50, threshold2=100)

        edges_pil = Image.fromarray(edges)  # Convert the NumPy array to a PIL image

        edges_pil.save('output_grid.png')
        width, height = image.size

        

    except IOError:
        print("Image (%s) could not be loaded." % filename)
        return
    bm = Bitmap(image, blacklevel=0.5)
    # bm.invert()
    plist = bm.trace(
        turdsize=2,
        turnpolicy=POTRACE_TURNPOLICY_MINORITY,
        alphamax=1,
        opticurve=False,
        opttolerance=0.2,
    )
    with open(f"{filename}.svg", "w") as fp:
       
        fp.write(
            f'''<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{image.width}" height="{image.height}" viewBox="0 0 {image.width} {image.height}">''')
        parts = []
        for curve in plist:
            fs = curve.start_point
            parts.append(f"M{fs.x},{fs.y}")
            for segment in curve.segments:
                if segment.is_corner:
                    a = segment.c
                    b = segment.end_point
                    parts.append(f"L{a.x},{a.y}L{b.x},{b.y}")
                else:
                    a = segment.c1
                    b = segment.c2
                    c = segment.end_point
                    parts.append(f"C{a.x},{a.y} {b.x},{b.y} {c.x},{c.y}")
            parts.append("z")
        fp.write(f'<path stroke="none" fill="black" fill-rule="evenodd" d="{"".join(parts)}"/>')
        fp.write("</svg>")




def image_to_svg(image_path, output_path, scale_factor=1, color_count=2, stroke_width=8):
    # Open image
    with Image.open(image_path) as img:
        width, height = img.size

        img = img.resize((width // scale_factor, height // scale_factor))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.quantize(colors=color_count).convert("RGB")

        grayscale_img = img.convert('L')






        gray_array = np.array(grayscale_img)

        gray_array = cv2.bilateralFilter(gray_array, d=9, sigmaColor=100, sigmaSpace=75)

        edges = cv2.Canny(gray_array, threshold1=50, threshold2=100)

        width, height = img.size


        # Begin SVG content
        svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * scale_factor}" height="{height * scale_factor}">\n'
        
        # Loop through image pixels
        for y in range(height):
            for x in range(width):
                if edges[y, x] > 0:  # Check if the pixel is part of an edge
                    # Get the color of the pixel for the stroke color
                    color = img.getpixel((x, y))
                    # Scale the rectangle size to show larger blocks
                    rect_x = x * scale_factor
                    rect_y = y * scale_factor
                    rect_width = scale_factor
                    rect_height = scale_factor
                    # Adding the stroke and removing the fill
                    svg_content += f'  <rect x="{rect_x}" y="{rect_y}" width="{rect_width}" height="{rect_height}" stroke="rgb{color}" stroke-width="{stroke_width}" fill="none" />\n'
        
        # End SVG content
        svg_content += '</svg>\n'
        
        # Write the SVG file
        with open(output_path, 'w') as f:
            f.write(svg_content)


def resize_image(image_path, size, border_size=10):
    """Resize image and add a transparent border."""
    img = Image.open(image_path)
    img = Image.fromarray(rembg.remove(np.array(img)))

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    if img.width > img.height:
        img = img.rotate(90, expand=True)
    
    img = img.resize(size, Image.Resampling.LANCZOS)
    
    img_with_border = ImageOps.expand(img, border=border_size, fill=(0, 0, 0, 0))  # RGBA fill (0, 0, 0, 0)

    return img_with_border




def create_grid(image_paths, grid_size, image_size, border_size=10):
    cols, rows = grid_size
    bordered_size = (image_size[0] + 2 * border_size, image_size[1] + 2 * border_size)
    
    grid_width = cols * bordered_size[0]
    grid_height = rows * bordered_size[1]

    grid_img = Image.new("RGB", (grid_width, grid_height))

    for i, img_path in enumerate(image_paths):
        img = resize_image(img_path, image_size, border_size)
        x = (i % cols) * bordered_size[0]
        y = (i // cols) * bordered_size[1]
        grid_img.paste(img, (x, y))

    return grid_img








img_counter = 0
image_files = []
for path in os.listdir("./image/"):
    path_in_str = './image/'+ str(path) 
    if img_counter < 6:
        img_counter += 1
        image_files.append(path_in_str)
    if img_counter == 6:
        img_counter = 0
        break

grid = create_grid(image_files, (3, 2), (1440, 1920))
##grid.show()
grid.save("output_grid.jpg")

#file_to_svg("./output_grid.jpg")
image_to_svg("./output_grid.jpg", "./output_image.svg")
print(image_files)
