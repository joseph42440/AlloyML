#!/usr/bin/python3

from src.GUI import start_GUI

model_paths = {
            'DoS': 'models/DoS_model_1598512707210.h5',
            'elongation': 'models/Elongation_model_1598512057434.h5',
            'tensile': 'models/Tensile_model_1598512362019.h5',
            'yield': 'models/Yield_model_1598512418828.h5'
        }

start_GUI(model_paths)
