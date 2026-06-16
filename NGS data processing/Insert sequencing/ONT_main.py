def ONT_main_gui():
    # Import packages used
    import os, glob
    import PySimpleGUI as sg

    #
    cols = [[sg.Button('Save reference sequence')],
            [sg.Button('Extract reads')],
            [sg.Button('Filter reads')],
                 [sg.Button('Perform MSA')],
                 [sg.Button('Generate consensus')]
                 ]

    layout = [[sg.Column(cols, element_justification='c', vertical_alignment="top")]]
    window_main = sg.Window('Nanopore consesus generator', layout, resizable=True)

    #
    while True:
        event, values = window_main.read()
        if event == sg.WIN_CLOSED:
            break
            window_main.close()

        if event == 'Save reference sequence':
            from ONT_ref_seq import ONT_ref_seq_main_gui

            ONT_ref_seq_main_gui()

        if event == 'Extract reads':
            from ONT_extract import ONT_extract_main_gui

            ONT_extract_main_gui()
        
        if event == 'Filter reads':
            from ONT_filter import ONT_filter_main_gui

            ONT_filter_main_gui()

        if event == 'Perform MSA':
            from ONT_MSA import ONT_MSA_main_gui

            ONT_MSA_main_gui()

        if event == 'Generate consensus':
            from ONT_cons import ONT_cons_main_gui

            ONT_cons_main_gui()

ONT_main_gui()