import cv2
import os

directory = 'D:/Final year project/Dataset_250'

if not os.path.exists(directory):
    os.mkdir(directory)

# Create directories for letters
for i in range(65, 91):
    letter = chr(i)
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')

# Create directories for digits
for i in range(48, 58):
    digit = chr(i)
    if not os.path.exists(f'{directory}/{digit}'):
        os.mkdir(f'{directory}/{digit}')

# Create directory for blank
if not os.path.exists(f'{directory}/blank'):
    os.mkdir(f'{directory}/blank')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    # Count files in each directory
    count_letters = {chr(i).lower(): len(os.listdir(f'{directory}/{chr(i)}')) for i in range(65, 91)}
    count_digits = {chr(j): len(os.listdir(f'{directory}/{chr(j)}')) for j in range(48, 58)}
    count_blank = len(os.listdir(os.path.join(directory, 'blank')))

    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    frame_roi = frame[40:300, 0:300]
    cv2.imshow("ROI", frame_roi)
    frame_resized = cv2.resize(frame_roi, (250, 250))

    interrupt = cv2.waitKey(25)
    if interrupt & 0xFF == 13:  # 13 is the ASCII code for 'Enter' key
        break
    
    if interrupt & 0xFF in range(48, 58):
        digit = chr(interrupt & 0xFF)
        cv2.imwrite(os.path.join(directory, digit, f'{count_digits[digit]+1}.jpg'), frame_resized)
        print(f"Saved image for {digit} with count {count_digits[digit]+1}")
        count_digits[digit] += 1
        
    
    elif interrupt & 0xFF in range(65, 91) or interrupt & 0xFF in range(97, 123):
        letter = chr(interrupt & 0xFF).upper()
        cv2.imwrite(os.path.join(directory, letter, f'{count_letters[letter.lower()]+1}.jpg'), frame_resized)
        print(f"Saved image for {letter} with count {count_letters[letter.lower()]+1}")
        count_letters[letter.lower()] += 1

    elif interrupt & 0xFF == ord('.'):
        cv2.imwrite(os.path.join(directory, 'blank', f'{count_blank+1}.jpg'), frame_resized)
        print(f"Saved image for BLANK with count {count_blank+1}")
        count_blank += 1

cap.release()
cv2.destroyAllWindows()
