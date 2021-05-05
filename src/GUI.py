#!/usr/bin/python3
import os
import sys
import re
import pickle
import threading
import shutil
import tkinter as tk
from tkinter import messagebox, Button, Label, Grid, Text, END, N, S, E, W, BOTH, LEFT, ttk, PhotoImage
import tkinter.filedialog as filedialog
from tensorflow import keras


from src.PredictOutput import predictOutput
from src.CompositionScan import scanSettings, compositionScan

model_paths = {}

def get_models():
    models = {}
    global models_paths
    for key, value in model_paths.items():
        models[key] = keras.models.load_model(value, compile=False)
    return models

class textRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        self.widget.configure(state="normal")
        self.widget.insert("end", text, (self.tag,))
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

    def flush(self):
        self.widget.configure(state="normal")
        self.widget.insert("end", '\n', (self.tag,))
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")

class sterileText(tk.Text):
    def __init__(self, *args, **kwargs):
        super(sterileText, self).__init__(*args, **kwargs)

    def getList(self):
        return list(map(int, re.findall(r'\d+', super().get('1.0', 'end'))))

    def getFloat(self):
        return float(re.findall(r'[-+]?\d*\.\d+|\d+', super().get('1.0', 'end'))[0])

    def getInt(self):
        return int(re.findall(r'[-+]?\d*\.\d+|\d+', super().get('1.0', 'end'))[0])

class scrollFrame(tk.Frame):

    def __init__(self, parent, rows, columns):
        super().__init__(parent,  borderwidth=0)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='news')
        self.scrollbar = tk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.inner = tk.Frame(self.canvas, borderwidth=0)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.inner.bind('<Configure>', self.resize)
        self.canvas.bind('<Configure>', self.frameWidth)
        self.num_rows = rows
        self.num_columns = columns

        for x in range(self.num_columns):
            Grid.columnconfigure(self.inner, x, weight=1)
        for y in range(self.num_rows):
            Grid.rowconfigure(self.inner, y, weight=1)

    def frameWidth(self, event):
        canvas_width = event.width
        self.canvas.itemconfig(self.window, width = canvas_width)

    def resize(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

class consoleWindow:
    def __init__(self, master):
        self.frame = tk.Toplevel(master)
        self.frame.iconbitmap('graphics/AlloyML.ico')
        self.frame.title('AlloyML Console')
        self.text = tk.Text(self.frame)
        self.scrollbar = tk.Scrollbar(self.frame, command=self.text.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.text['yscrollcommand'] = self.scrollbar.set
        self.text.pack(fill='both', expand=True)
        sys.stdout = textRedirector(self.text, "stdout")
        sys.stderr = textRedirector(self.text, "stderr")

class mainMenu:
    def __init__(self, master):
        self.master = master
        self.master.geometry("270x270")
        self.frame = tk.Frame(self.master)
        self.prediction_dataset = None

        Label(self.frame, text="Main Menu", font='Helvetica 10 bold').grid(row=0, column=1, columnspan=3, sticky=N + S + E + W, pady=3)
        Label(self.frame, text="Mode:").grid(row=1, column=1, sticky=N + S + E + W, pady=3)
        self.mode_box = ttk.Combobox(self.frame, state="readonly",
                                values=[
                                    "DoS",
                                    "Mechanical"])
        self.mode_box.current(0)
        self.mode_box.grid(row=1, column=2, columnspan=2, stick=N + S + E + W, pady=3)

        Button(self.frame, text='Composition Scan', command=self.compositionFrame).grid(row=2, column=1, columnspan=3, rowspan=2, sticky=N + S + E + W, pady=3)
        Button(self.frame, text="Predict Output", command=self.openPredictionThread).grid(row=4, column=1, columnspan=3, rowspan=2, sticky=N + S + E + W, pady=3)

        # Formatting
        for x in range(5):
            Grid.columnconfigure(self.frame, x, weight=1)
        for y in range(7):
            Grid.rowconfigure(self.frame, y, weight=1)
        self.frame.pack(anchor=N, fill=BOTH, expand=True, side=LEFT)

    def compositionFrame(self):
        if threading.active_count() != 1:
            tk.messagebox.showerror('Error', 'Please Wait For Process To Finish')
        else:
            compositionScanFrame(self.master, scanSettings(self.mode_box.get()))
            self.frame.destroy()

    def runThreadedPrediction(self):
        my_predictor = predictOutput(self.prediction_dataset, get_models(), self.mode_box.get())
        my_predictor.run()
        for folder in os.listdir('models/'):
            if os.path.isdir('models/' + folder):
                shutil.rmtree('models/' + folder)


    def openPredictionThread(self):
        if threading.active_count() != 1:
            tk.messagebox.showerror('Error', 'Please Wait For Process To Finish')
        else:
            try:
                os.mkdir('prediction_datasets/')
            except FileExistsError:
                pass
            try:
                os.mkdir('prediction_datasets/' + self.mode_box.get())
            except FileExistsError:
                pass
            file_types = ['Dataset', '.csv'],
            filename = filedialog.askopenfilename(initialdir=os.getcwd() + '/prediction_datasets/' + self.mode_box.get(),
                                                       title='Select Dataset',
                                                       filetypes=file_types, defaultextension='.csv')
            try:
                self.prediction_dataset = filename
                if self.prediction_dataset is None or self.prediction_dataset == '':
                    tk.messagebox.showerror('Error', 'Unable To Load Dataset ''')
                else:
                    consoleWindow(self.master)
                    threading.Thread(target=self.runThreadedPrediction).start()
            except Exception:
                tk.messagebox.showerror('Error Loading Dataset', 'Unable To Open File: %r' % filename)

class compositionScanFrame:
    def captureConfig(self):
        self.settings = scanSettings(self.settings.mode)

        for index, key in enumerate(self.settings.categorical_inputs.keys()):
            self.settings.categorical_inputs[key] = self.categorical_values[index].getList()
            if not self.settings.categorical_inputs[key]:
                tk.messagebox.showerror('Error Saving Configuration', 'All Categories Need At Least One Option')
                return False

        for index, key in enumerate(self.settings.range_based_inputs.keys()):
            self.settings.range_based_inputs[key] = [self.range_based_min[index].getFloat(),
                                                    self.range_based_max[index].getFloat()]

        for index, key in enumerate(self.settings.targets.keys()):
            self.settings.targets[key] = self.scan_settings_values[index].getInt()

        self.settings.max_steps = self.scan_settings_values[-1].getInt()
        if self.settings.max_steps < 1:
            tk.messagebox.showerror('Error Saving Configuration', 'Max Steps Must Be A Positive Integer')
            return False
        return True

    def loadSettings(self):
        try:
            os.mkdir('scan_settings/')
        except FileExistsError:
            pass
        try:
            os.mkdir('scan_settings/' + self.settings.mode)
        except FileExistsError:
            pass
        file_types = ['Configuration File', '.pkl'],
        filename = filedialog.askopenfilename(initialdir=os.getcwd() + '/scan_settings/' + self.settings.mode,
                                                   title='Select Configuration',
                                                   filetypes=file_types, defaultextension='.pkl')
        try:
            global models
            compositionScanFrame(self.master, settings=pickle.load(open(filename, 'rb')))
            self.frame.destroy()
        except Exception:
            tk.messagebox.showerror('Error Loading Configuration', 'Unable to open file: %r' % filename)

    def saveSettings(self):
        try:
            os.mkdir('scan_settings/')
        except FileExistsError:
            pass
        try:
            os.mkdir('scan_settings/' + self.settings.mode)
        except FileExistsError:
            pass
        if self.captureConfig():
            file_types = ['Configuration File', '.pkl'],
            filename = filedialog.asksaveasfilename(initialdir=os.getcwd() + '/scan_settings/' + self.settings.mode, title='Save Configuration',
                                                    filetypes=file_types, defaultextension='.pkl')
        try:
            pickle.dump(self.settings, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)
        except Exception:
            tk.messagebox.showerror('Error Saving Configuration', 'Unable to save file: %r' % filename)

    def runThreadedScan(self):
        my_composition_scan = compositionScan(self.settings, get_models())
        my_composition_scan.run()
        for folder in os.listdir('models/'):
            if os.path.isdir('models/' + folder):
                shutil.rmtree('models/' + folder)

    def openScanThread(self):
        if self.captureConfig():
            if threading.active_count() != 1:
                tk.messagebox.showerror('Error', 'Please Wait For Process To Finish')
            else:
                consoleWindow(self.master)
                threading.Thread(target=self.runThreadedScan).start()

    def close(self):
        if threading.active_count() != 1:
            tk.messagebox.showerror('Error', 'Please Wait For Process To Finish')
        else:
            self.frame.destroy()
            mainMenu(self.master)

    def getBackgroundColor(self, index):
        if index % 2:
            return 'alice blue'
        else:
            return 'white smoke'

    def __init__(self, master, settings):
        self.master = master
        self.settings = settings
        self.master.geometry("700x500")
        self.frame = tk.Frame(self.master)
        self.num_rows = 28
        self.range_based_min = []
        self.range_based_max = []
        self.categorical_values = []
        self.scan_settings_values = []

        # Range-Based Constraints
        Label(self.frame, text="Range-Based Inputs", font='Helvetica 10 bold').grid(row=0, column=0, pady=3)
        self.range_based_frame = scrollFrame(self.frame, len(self.settings.range_based_inputs.keys()), 4)
        self.range_based_frame.grid(row=1, column=0, rowspan=self.num_rows - 1, sticky=N + S + E + W, pady=3)

        for index, key in enumerate(self.settings.range_based_inputs.keys()):
            background = self.getBackgroundColor(index)
            Label(self.range_based_frame.inner, bg=background, text=key + " :").grid(row=index, column=0, pady=3, sticky='ew')
            Label(self.range_based_frame.inner, bg=background, text="to").grid(row=index, column=2, sticky='ew')
            self.range_based_min.append(sterileText(self.range_based_frame.inner, height=1, width=6))
            self.range_based_min[index].insert(END, self.settings.range_based_inputs[key][0])
            self.range_based_min[index].grid(row=index, column=1, sticky='ew')
            self.range_based_max.append(sterileText(self.range_based_frame.inner, height=1, width=6))
            self.range_based_max[index].insert(END, self.settings.range_based_inputs[key][1])
            self.range_based_max[index].grid(row=index, column=3, sticky='ew')

        # Categorical Constraints
        Label(self.frame, text="Categorical Inputs", font='Helvetica 10 bold').grid(row=0, column=1, pady=3)
        self.categorical_frame = scrollFrame(self.frame, len(self.settings.categorical_inputs.keys()), 2)
        self.categorical_frame.grid(row=1, column=1, rowspan=int((self.num_rows - 1) / 2) - 1, sticky=N + S + E + W, pady=3)

        for index, key in enumerate(self.settings.categorical_inputs.keys()):
            background = self.getBackgroundColor(index)
            Label(self.categorical_frame.inner, bg=background, text=key + " :").grid(row=index, column=0, pady=3, sticky='ew')
            self.categorical_values.append(sterileText(self.categorical_frame.inner, height=1, width=14))
            self.categorical_values[index].insert(END, str(self.settings.categorical_inputs[key]).replace(" ", ""))
            self.categorical_values[index].grid(row=index, column=1, sticky='ew')

        # Scan Settings
        Label(self.frame, text="Scan Settings", font='Helvetica 10 bold').grid(row=int(self.num_rows/2), column=1, pady=3, sticky='ew')
        self.scan_settings_frame = scrollFrame(self.frame, len(self.settings.range_based_inputs.keys()), 2)
        self.scan_settings_frame.grid(row=int(self.num_rows / 2) + 1, column=1, rowspan=int((self.num_rows - 1) / 2), sticky=N + S + E + W, pady=3)

        for index, key in enumerate(self.settings.targets.keys()):
            background = self.getBackgroundColor(index)
            Label(self.scan_settings_frame.inner, bg=background, text='target ' + key + " :").grid(row=index, column=0, pady=3, sticky='ew')
            self.scan_settings_values.append(sterileText(self.scan_settings_frame.inner, height=1, width=14))
            self.scan_settings_values[index].insert(END, str(self.settings.targets[key]))
            self.scan_settings_values[index].grid(row=index, column=1, sticky='ew')
        index = len(self.settings.targets.keys())
        background = self.getBackgroundColor(index)
        Label(self.scan_settings_frame.inner, bg=background, text='max steps : ').grid(row=index, column=0, pady=3, sticky='ew')
        self.scan_settings_values.append(sterileText(self.scan_settings_frame.inner, height=1, width=14))
        self.scan_settings_values[index].insert(END, str(self.settings.max_steps))
        self.scan_settings_values[index].grid(row=index, column=1, sticky='ew')

        # Legend
        Label(self.frame, text="Legend", font='Helvetica 10 bold').grid(row=0, column=2, pady=0, sticky='ew')
        if self.settings.mode == 'DoS':
            Label(self.frame, text="Recrystallised").grid(row=1, column=2, pady=0, sticky='ew')
            Label(self.frame, text="0 = False").grid(row=2, column=2, pady=0, sticky='w')
            Label(self.frame, text="1 = True").grid(row=3, column=2, pady=0, sticky='w')
            Label(self.frame, text="Temper").grid(row=4, column=2, pady=0, sticky='ew')
            Label(self.frame, text="1 = H (lab)").grid(row=5, column=2, pady=0, sticky='w')
            Label(self.frame, text="2 = H116").grid(row=6, column=2, pady=0, sticky='w')
            Label(self.frame, text="3 = H131").grid(row=7, column=2, pady=0, sticky='w')
            Label(self.frame, text="4 = H321").grid(row=8, column=2, pady=0, sticky='w')
            Label(self.frame, text="5 = O").grid(row=9, column=2, pady=0, sticky='w')
            Label(self.frame, text="6 = Stabilised").grid(row=10, column=2, pady=0, sticky='w')
            Label(self.frame, text="7 = Unknown").grid(row=11, column=2, pady=0, sticky='w')
        elif self.settings.mode == 'Mechanical':
            Label(self.frame, text="Processing Condition").grid(row=1, column=2, pady=0, sticky='ew')
            Label(self.frame, text="1 = As-cast or as-fabricated").grid(row=2, column=2, pady=0, sticky='w')
            Label(self.frame, text="2 = Annealed, solutionised").grid(row=3, column=2, pady=0, sticky='w')
            Label(self.frame, text="3 = H (soft)").grid(row=4, column=2, pady=0, sticky='w')
            Label(self.frame, text="4 = H (hard)").grid(row=5, column=2, pady=0, sticky='w')
            Label(self.frame, text="5 = T1").grid(row=6, column=2, pady=0, sticky='w')
            Label(self.frame, text="6 = T2").grid(row=7, column=2, pady=0, sticky='w')
            Label(self.frame, text="7 = T3 (incl. T341 etc.)").grid(row=8, column=2, pady=0, sticky='w')
            Label(self.frame, text="8 = T4").grid(row=9, column=2, pady=0, sticky='w')
            Label(self.frame, text="9 = T5").grid(row=10, column=2, pady=0, sticky='w')
            Label(self.frame, text="10 = T6 (incl. T651 etc.)").grid(row=11, column=2, pady=0, sticky='w')
            Label(self.frame, text="11 = T7 (incl. T777 etc.)").grid(row=12, column=2, pady=0, sticky='w')
            Label(self.frame, text="12 = T8 (incl. T851 etc.)").grid(row=13, column=2, pady=0, sticky='w')
            Label(self.frame, text="13 = Lab routine or unknown").grid(row=14, column=2, pady=0, sticky='w')



        # Buttons
        Button(self.frame, text="Run", command=self.openScanThread).grid(row=self.num_rows-9, column=2,  rowspan=2, sticky=N+S+E+W, pady=3)
        Button(self.frame, text="Load Configuration", command=self.loadSettings).grid(row=self.num_rows-7, column=2, rowspan=2, sticky=N + S + E + W, pady=3)
        Button(self.frame, text="Save Configuration", command=self.saveSettings).grid(row=self.num_rows-5, column=2, rowspan=2, sticky=N + S + E + W, pady=3)
        Button(self.frame, text="Main Menu", command=self.close).grid(row=self.num_rows-3, column=2,  rowspan=2, sticky=N+S+E+W, pady=3)

        # Formatting
        Grid.columnconfigure(self.frame, 0, weight=1)
        Grid.columnconfigure(self.frame, 1, weight=1)
        Grid.columnconfigure(self.frame, 2, weight=0)
        Grid.columnconfigure(self.frame, 3, weight=0)
        for y in range(self.num_rows):
            Grid.rowconfigure(self.frame, y, weight=1)
        self.frame.pack(anchor=N, fill=BOTH, expand=True, side=LEFT)


def start_GUI(paths):
    global model_paths
    model_paths = paths
    root = tk.Tk()
    root.iconbitmap('graphics/AlloyML.ico')
    root.title('AlloyML v2.4.0')
    app = mainMenu(root)
    root.mainloop()