import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

def YOLO_Interface(image_path):
    # 1. Load your trained YOLO model
    model = YOLO('/kaggle/input/placeholder/other/default/1/best (1).pt') # Replace with your model path
    
    # 3. Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        exit()
    
    # Convert BGR to RGB for consistent display with matplotlib later
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = image_rgb.copy() # Create a copy to draw on
    
    # 4. Perform prediction (no drawing with .plot() here, just get raw results)
    # We set save=False and show=False because we'll handle drawing manually
    results = model.predict(source=image_path, conf=0.5, iou=0.9, save=False, show=False)
    
    # 5. Process results and draw boxes manually using OpenCV
    for r in results:
        if r.boxes: # Check if any bounding boxes were detected
            for box in r.boxes:
                # Get coordinates (xyxy format: xmin, ymin, xmax, ymax)
                # .cpu().numpy() converts the tensor to a NumPy array on the CPU
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
    
                # Get class ID and confidence
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = model.names[class_id]
    
                
                color = (0, 255, 0) 
                if model.task == 'segment': # Example: different color for segmentation if applicable
                    color = (255, 0, 255) # Magenta
    
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2) 
    
                # Create the label text
                label = f"{class_name} {confidence:.2f}"
    
                # Calculate text size and position to avoid overlap
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    
                # Position text above the box, or inside if space is limited
                text_x = x1
                text_y = y1 - 10 # 10 pixels above the top of the box
    
                # Ensure text is not off-image at the top
                if text_y < 0:
                    text_y = y1 + text_size[1] + 10 # Place below if not enough space above
    
                # Draw background rectangle for text for better readability
                text_bg_color = (0, 0, 0) # Black background
                #cv2.rectangle(display_image, (text_x, text_y - text_size[1] - 5), # top-left
                 #             (text_x + text_size[0] + 5, text_y + 5), # bottom-right
                  #            text_bg_color, -1) # -1 fills the rectangle
    
                #cv2.putText(display_image, label, (text_x + 2, text_y), font,
                 #           font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA) # White text
    
    # 6. Display the annotated image using Matplotlib
    plt.figure(figsize=(12, 10)) # Adjust figure size for better viewing
    plt.imshow(display_image)
    plt.axis('off') # Hide axes
    plt.title('Model Prediction with Custom Labels')
    plt.show()

YOLO_Interface()

import os
import cv2
from ultralytics import YOLO

# Load YOLO model once
model = YOLO('/kaggle/input/placeholder/other/default/1/best (1).pt')

def YOLO_cropper(image_path, output_folder="/kaggle/working/cropped_words", row_tolerance=20):
    """
    Detect words in an image using YOLO, crop them, and save in left-to-right, top-to-bottom order.
    
    Args:
        image_path (str): Path to the input image.
        output_folder (str): Folder where crops will be saved.
        row_tolerance (int): Pixel tolerance for grouping bounding boxes into the same row.
    
    Returns:
        list: A list of file paths to the saved cropped images in reading order.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Run YOLO
    results = model(image_path)

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

    # Collect box info
    crops_info = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        crops_info.append((i, x1, y1, x2, y2))

    # Sort: first by row (y1), then by column (x1)
    crops_info.sort(key=lambda b: (b[2] // row_tolerance, b[1]))

    # Load original image
    image = cv2.imread(image_path)

    saved_paths = []
    for idx, (i, x1, y1, x2, y2) in enumerate(crops_info, start=1):
        crop = image[y1:y2, x1:x2]
        save_path = os.path.join(output_folder, f"word_{idx}.jpg")
        cv2.imwrite(save_path, crop)
        saved_paths.append(save_path)

    print(f"✅ Saved {len(saved_paths)} crops in reading order to '{output_folder}'")
    return saved_paths
x = YOLO_cropper('/kaggle/input/dataseet/data/images/test/aahw_0007.jpg')
for i in x:
    print(i)

import torch
import torch.nn as nn

# Same CRNN definition as in training
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc,64,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True), nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True)
        )
        self.rnn1 = nn.LSTM(512, nh, bidirectional=True, dropout=0.3)
        self.rnn2 = nn.LSTM(nh*2, nh, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(2, 0, 1)
        recurrent, _ = self.rnn1(conv)
        recurrent, _ = self.rnn2(recurrent)
        output = self.fc(recurrent)
        return output.permute(1, 0, 2)

  # Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate model with same hyperparameters
model = CRNN(imgH=32, nc=1, nclass= 303, nh=256).to(DEVICE)

# Load weights
checkpoint = torch.load("/kaggle/input/crnn-amharic-hand-writing-detector/keras/default/1/best_crnn (1).pth", map_location=DEVICE)
model.load_state_dict(checkpoint)

model.eval()
print("Model loaded and ready!")

import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    new_w = max(int(32 * (w / h)), 32)
    img = cv2.resize(img, (new_w, 32))
    img = (img / 255.0).astype(np.float32)
    img = (img - 0.5) / 0.5  # normalize
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [B,C,H,W]
    return img.to(DEVICE)

BLANK = 302
def ctc_greedy_decoder(output, idx_to_char, blank=BLANK):
    # output: [B, T, nclass]
    preds = output.softmax(2).argmax(2)  # [B, T]
    preds = preds[0].cpu().numpy().tolist()

    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:  # collapse repeats, ignore blanks
            decoded.append(idx_to_char.get(p, "�"))
        prev = p
    return "".join(decoded)
  amharic_mapping = {
    # Basic consonants + vowels
    0: 'ሀ', 1: 'ሁ', 2: 'ሂ', 3: 'ሃ', 4: 'ሄ', 5: 'ህ', 6: 'ሆ',
    7: 'ለ', 8: 'ሉ', 9: 'ሊ', 10: 'ላ', 11: 'ሌ', 12: 'ል', 13: 'ሎ', 14: 'ሏ',
    15: 'ሐ', 16: 'ሑ', 17: 'ሒ', 18: 'ሓ', 19: 'ሔ', 20: 'ሕ', 21: 'ሖ', 22: 'ሗ',
    23: 'መ', 24: 'ሙ', 25: 'ሚ', 26: 'ማ', 27: 'ሜ', 28: 'ም', 29: 'ሞ', 30: 'ሟ',
    31: 'ሠ', 32: 'ሡ', 33: 'ሢ', 34: 'ሣ', 35: 'ሤ', 36: 'ሥ', 37: 'ሦ', 38: 'ሧ',
    39: 'ረ', 40: 'ሩ', 41: 'ሪ', 42: 'ራ', 43: 'ሬ', 44: 'ር', 45: 'ሮ', 46: 'ሯ',
    47: 'ሰ', 48: 'ሱ', 49: 'ሲ', 50: 'ሳ', 51: 'ሴ', 52: 'ስ', 53: 'ሶ', 54: 'ሷ',
    55: 'ሸ', 56: 'ሹ', 57: 'ሺ', 58: 'ሻ', 59: 'ሼ', 60: 'ሽ', 61: 'ሾ', 62: 'ሿ',
    63: 'ቀ', 64: 'ቁ', 65: 'ቂ', 66: 'ቃ', 67: 'ቄ', 68: 'ቅ', 69: 'ቆ', 70: 'ቋ',
    71: 'በ', 72: 'ቡ', 73: 'ቢ', 74: 'ባ', 75: 'ቤ', 76: 'ብ', 77: 'ቦ', 78: 'ቧ',
    79: 'ቨ', 80: 'ቩ', 81: 'ቪ', 82: 'ቫ', 83: 'ቬ', 84: 'ቭ', 85: 'ቮ', 86: 'ቯ',
    87: 'ተ', 88: 'ቱ', 89: 'ቲ', 90: 'ታ', 91: 'ቴ', 92: 'ት', 93: 'ቶ', 94: 'ቷ',
    95: 'ቸ', 96: 'ቹ', 97: 'ቺ', 98: 'ቻ', 99: 'ቼ', 100: 'ች', 101: 'ቾ', 102: 'ቿ',
    103: 'ኀ', 104: 'ኁ', 105: 'ኂ', 106: 'ኃ', 107: 'ኄ', 108: 'ኅ', 109: 'ኆ',
    110: 'ነ', 111: 'ኑ', 112: 'ኒ', 113: 'ና', 114: 'ኔ', 115: 'ን', 116: 'ኖ', 117: 'ኗ',
    118: 'ኘ', 119: 'ኙ', 120: 'ኚ', 121: 'ኛ', 122: 'ኜ', 123: 'ኝ', 124: 'ኞ', 125: 'ኟ',
    126: 'አ', 127: 'ኡ', 128: 'ኢ', 129: 'ኣ', 130: 'ኤ', 131: 'እ', 132: 'ኦ', 133: 'ኧ',
    134: 'ከ', 135: 'ኩ', 136: 'ኪ', 137: 'ካ', 138: 'ኬ', 139: 'ክ', 140: 'ኮ', 141: 'ኯ',
    142: 'ኸ', 143: 'ኹ', 144: 'ኺ', 145: 'ኻ', 146: 'ኼ', 147: 'ኽ', 148: 'ኾ', 149: 'ዀ', 150: 'ዃ',
    151: 'ወ', 152: 'ዉ', 153: 'ዊ', 154: 'ዋ', 155: 'ዌ', 156: 'ው', 157: 'ዎ', 158: 'ዏ',
    159: 'ዐ', 160: 'ዑ', 161: 'ዒ', 162: 'ዓ', 163: 'ዔ', 164: 'ዕ', 165: 'ዖ',
    166: 'ዘ', 167: 'ዙ', 168: 'ዚ', 169: 'ዛ', 170: 'ዜ', 171: 'ዝ', 172: 'ዞ', 173: 'ዟ',
    174: 'ዠ', 175: 'ዡ', 176: 'ዢ', 177: 'ዣ', 178: 'ዤ', 179: 'ዥ', 180: 'ዦ', 181: 'ዧ',
    182: 'የ', 183: 'ዩ', 184: 'ዪ', 185: 'ያ', 186: 'ዬ', 187: 'ይ', 188: 'ዮ',
    189: 'ደ', 190: 'ዱ', 191: 'ዲ', 192: 'ዳ', 193: 'ዴ', 194: 'ድ', 195: 'ዶ', 196: 'ዷ',
    197: 'ጀ', 198: 'ጁ', 199: 'ጂ', 200: 'ጃ', 201: 'ጄ', 202: 'ጅ', 203: 'ጆ', 204: 'ጇ',
    205: 'ገ', 206: 'ጉ', 207: 'ጊ', 208: 'ጋ', 209: 'ጌ', 210: 'ግ', 211: 'ጎ', 212: 'ጏ',
    213: 'ጠ', 214: 'ጡ', 215: 'ጢ', 216: 'ጣ', 217: 'ጤ', 218: 'ጥ', 219: 'ጦ', 220: 'ጧ',
    221: 'ጨ', 222: 'ጩ', 223: 'ጪ', 224: 'ጫ', 225: 'ጬ', 226: 'ጭ', 227: 'ጮ', 228: 'ጯ',
    229: 'ጰ', 230: 'ጱ', 231: 'ጲ', 232: 'ጳ', 233: 'ጴ', 234: 'ጵ', 235: 'ጶ', 236: 'ጷ',
    237: 'ጸ', 238: 'ጹ', 239: 'ጺ', 240: 'ጻ', 241: 'ጼ', 242: 'ጽ', 243: 'ጾ', 244: 'ጿ',
    245: 'ፀ', 246: 'ፁ', 247: 'ፂ', 248: 'ፃ', 249: 'ፄ', 250: 'ፅ', 251: 'ፆ', 252: 'ፇ',
    253: 'ፈ', 254: 'ፉ', 255: 'ፊ', 256: 'ፋ', 257: 'ፌ', 258: 'ፍ', 259: 'ፎ', 260: 'ፏ',
    261: 'ፐ', 262: 'ፑ', 263: 'ፒ', 264: 'ፓ', 265: 'ፔ', 266: 'ፕ', 267: 'ፖ', 268: 'ፗ',

    # Special characters / punctuation / numbers
    269: "!", 270: ":-", 271: "<", 272: "(", 273: "«", 274: "፥", 275: "%", 276: "»", 277: ")",
    278: ">", 279: ".", 280: "+", 281: "፣", 282: "-", 283: "።", 284: "/", 
    285: "0", 286: "1", 287: "2", 288: "3", 289: "4", 290: "5", 291: "6", 292: "7", 293: "8", 294: "9", 
    295: "፡", 296: "፤", 297: "...", 298: "*", 299: "#", 300: "?"
}

import matplotlib.pyplot as plt
def CNNR_Interface(img_path):
    img = preprocess_image(img_path)
    ig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(ig)
    plt.axis('off') # Turn off axis labels and ticks
    plt.title(img_path)
    plt.show()
    with torch.no_grad():
        output = model(img)  # [B, T, nclass]
    
    predicted_text = ctc_greedy_decoder(output, amharic_mapping, blank=BLANK)
    print("Predicted:", predicted_text)
    return predicted_text

CNNR_Interface("/kaggle/working/cropped_words/word_1.jpg")

import os
import shutil

def clear_folder(folder_path):
    """
    Ensures the folder exists, then deletes all files and subdirectories inside it.
    
    Args:
        folder_path (str): Path to the folder to clear.
    """
    # Create folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created folder '{folder_path}'")
        return

    # Clear existing contents
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove subdirectory
        except Exception as e:
            print(f"⚠️ Failed to delete {file_path}. Reason: {e}")
    
    print(f"✅ Cleared all contents in '{folder_path}'")

def pipeline(img_path):
    detected_text = ''
    clear_folder('/kaggle/working/cropped_words')
    print('done cleaning')
    YOLO_Interface(img_path)
    print('done locating')
    locs = YOLO_cropper(img_path)
    print('done cropping')
    
    if len(locs) == 0:
        return detected_text
    for loc in locs:
        print(loc)
        detected_text += CNNR_Interface(loc)
    
    return detected_text

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
print(pipeline('/kaggle/input/dataseet/data/images/test/aahw_0007.jpg'))
 
