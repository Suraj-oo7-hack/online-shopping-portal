from google.colab import files

# Upload image
uploaded = files.upload()
if uploaded:
    file_name = list(uploaded.keys())[0]
    image = Image.open(file_name)
    
    # Generate Ghibli art
    strength = 0.6  # Adjust strength as needed
    ghibli_image = generate_ghibli_image(image, pipe, strength)
    
    # Display the result
    plt.imshow(ghibli_image)
    plt.axis('off')
    plt.show()

import tkinter as tk
from tkinter import filedialog

def upload_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open file dialog
    print(f"File selected: {file_path}")
    return file_path

import tkinter as tk
from tkinter import filedialog

def upload_file(""):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open file dialog
    print(f"File selected: {file_path}")
    return file_path


