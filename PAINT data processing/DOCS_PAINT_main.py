def process_main_gui():
    # Import packages used
    import os, glob
    import PySimpleGUI as sg

    #
    cols = [[sg.Button('Crop data')],
                 [sg.Button('Segment data')],
                 [sg.Button('Extract site localizations')],
                 [sg.Button('Process extracted data')]
                 ]

    layout = [[sg.Column(cols, element_justification='c', vertical_alignment="top")]]
    window_main = sg.Window('PAINT data processing for Barcode project', layout, resizable=True)

    #
    while True:
        event, values = window_main.read()
        if event == sg.WIN_CLOSED:
            break
            window_main.close()

        if event == 'Crop data':
            from DOCS_PAINT_data_crop import PAINT_crop_main_gui

            PAINT_crop_main_gui()

        if event == 'Segment data':
            from DOCS_PAINT_data_segmentation import PAINT_segment_main_gui

            PAINT_segment_main_gui()

        if event == 'Extract site localizations':
            from DOCS_PAINT_data_site_loc_extract import PAINT_func_proc_main_gui

            PAINT_func_proc_main_gui()

        if event == 'Process extracted data':
            from DOCS_PAINT_data_result_vis import PAINT_func_res_plot_main_gui

            PAINT_func_res_plot_main_gui()

process_main_gui()