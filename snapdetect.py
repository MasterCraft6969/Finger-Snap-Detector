import pyaudio
import scipy.fftpack as sf
import numpy as np
import time
import keyboard
import threading

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 2**10
COOLDOWN_TIME = 5  
IGNORE_TIME = 3

class SnappingDetector(object):

    def __init__(self):
        self.x = sf.fftfreq(CHUNK, 1.0 / RATE)[:int(CHUNK / 2)]
        self.audio = pyaudio.PyAudio()
        self.preDetect = -1
        self.lastMeans = [0]
        self.last_trigger_time = 0  
        self.detecting = True  
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        default_device_index = self.audio.get_default_input_device_info()['index']

        self.stream = self.audio.open(
            input=True,
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input_device_index=default_device_index,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback)
        print("Starting detection with default microphone... (Press 'p' to pause/resume detection)")
        self.stream.start_stream()

    def callback(self, in_data, frame_count, time_info, status):
        if not self.detecting:
            return (None, pyaudio.paContinue)

        if time.time() - self.start_time < IGNORE_TIME:
            return (None, pyaudio.paContinue)

        result = sf.fft(np.frombuffer(in_data, dtype=np.int16))
        self.y = np.abs(result)[:int(CHUNK / 2)]
        mean = np.mean(self.y)
        var = np.var(self.y)
        meansMean = np.mean(self.lastMeans)
        freqMean = np.mean(self.x * self.y) / mean

        if self.preDetect == -1:
            if 8000 < freqMean < 12000 and 10.0 * meansMean < mean and 10000 < mean and 200000000 < var:
                self.preDetect = mean
                self.preDetectTime = time.time()
        elif self.preDetectTime + 0.2 < time.time():
            if mean < self.preDetect:
                print('Snap detected! Checking cooldown...')
                self.send_key()
            self.preDetect = -1

        if 10 <= len(self.lastMeans):
            self.lastMeans.pop(0)
        self.lastMeans.append(mean)

        return (None, pyaudio.paContinue)

    def send_key(self):
        current_time = time.time()
        
        if current_time - self.last_trigger_time >= COOLDOWN_TIME:
            keyboard.press_and_release('space')
            print('Key sent! Cooldown started.')
            self.last_trigger_time = current_time
        else:
            print('In cooldown period. Key not sent.')

    def toggle_detection(self):
        self.detecting = not self.detecting
        if self.detecting:
            print("Resumed detection.")
        else:
            print("Paused detection.")

def keyboard(detector):
    while True:
        if keyboard.is_pressed('c'):
            print("\nExiting...")
            break
        if keyboard.is_pressed('p'):
            detector.toggle_detection()
            time.sleep(0.5)  

detector = SnappingDetector()
detect_thread = threading.Thread(target=detector.start)
control_thread = threading.Thread(target=keyboard, args=(detector,))

detect_thread.start()
control_thread.start()

detect_thread.join()
control_thread.join()
