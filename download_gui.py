import PySimpleGUI as sg
import threading
from diffusers import StableDiffusionPipeline
import time
import torch

class LoadingBar():
    def __init__(self):
        layout = [[sg.Text('Download is in progress, please wait before generating any images')],
                [sg.ProgressBar(max_value=100, orientation='h', size=(40,40), key='load_percent')]]
        
        self.main_window = sg.Window('Downloading...', layout, finalize=True)
        current_value = 0
        self.main_window['load_percent'].update(current_value)

    def start(self, model_path, dtype, variant):
        self.stop_event = threading.Event()
        func_thread = threading.Thread(target=self.counter,
                        args=(self.main_window, ),
                        daemon=True)
        func_thread.start()

        download_thread = threading.Thread(target=self.download_model, args= (model_path, dtype, variant,),  daemon=True)
        download_thread.start()
        i = 0
        while True:
            self.window, event, values = sg.read_all_windows()
            # event, values = self.window.read()#read_all_windows()
            if event == 'Exit':
                break
            elif event is None:
                continue
            elif event.startswith('update_'):
                # print(f'event: {event}, value: {values[event]}')
                key_to_update = event[len('update_'):]
                self.window[key_to_update].update(values[event])
                self.window.refresh()
            if not download_thread.is_alive():
                self.window['load_percent'].update(100)
                time.sleep(2)
                self.window.write_event_value('Exit', '')

                
                
        self.main_window.close()

    def download_model(self, model_path, dtype, variant):
        temp = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32, variant=variant)
        temp = None
        self.stop_event.set()


    def counter(self, window):
        import time
        import random
        if self.stop_event.is_set():
            return
        while True:
            for i in range(100):
                time.sleep(.1)
                try:
                    window.write_event_value('update_load_percent', i)
                except:
                    pass
   