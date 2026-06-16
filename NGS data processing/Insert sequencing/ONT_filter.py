# Helper method to draw figure from PysimpleGUI Demo (from: https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Pyplot_Bar_Chart.py)
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    #
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import tkinter as Tk
    from matplotlib.backend_bases import key_press_handler

    class Toolbar(NavigationToolbar2Tk):
        # Note the matplot tk canvas import
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        # only display the buttons we need
        toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                     t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

        def __init__(self, *args, **kwargs):
            super(Toolbar, self).__init__(*args, **kwargs)

    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side=Tk.RIGHT, fill=Tk.BOTH, expand=1)

    def on_key_press(event):
        key_press_handler(event, canvas, toolbar)
        canvas.TKCanvas.mpl_connect("key_press_event", on_key_press)

    return

# Function for read thresholding based on alignment score to reference
def read_filter(df_exp, exp_folder):
    # import packages used
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy import stats
    from scipy.signal import argrelmax
    import random
    import pandas as pd
    import PySimpleGUI as sg
    import numpy as np
    import csv

    # For checking size distribution in sample
    data_df = pd.read_hdf(df_exp, key='read_l')
    read_length_l = data_df['Read length'].values.tolist()

    # VARS CONSTS:
    _VARS = {'window': False}

    # \\  -------- PYSIMPLEGUI -------- //
    sg.ChangeLookAndFeel('Reddit')
    layout = [
        [sg.Frame('Controls:',
                  [[sg.Canvas(key='controls_cv')], [sg.B('Replot with default values')]])],
        [sg.Frame('Plot:',
                  [
                      [sg.Canvas(key='fig_cv',
                                 # it's important that you set this size
                                 size=(400 * 2, 400)
                                 )]
                  ])],
        [sg.Frame('Plot reformating:',
                  [
                      [sg.Text('Bins', size=(25, 1)),
                       sg.InputText(key='bins', size=(8, 1))],
                      [sg.Checkbox('Normalized', size=(30, 1), key='chk_norm')],
                      [sg.Text('Plot size in inches:', size=(25, 1)),
                       sg.InputText('width', key='w', size=(8, 1)), sg.InputText('height', key='h', size=(8, 1))],
                      [sg.Text('X limits', size=(25, 1)),
                       sg.InputText('min', key='x_min', size=(8, 1)), sg.InputText('max', key='x_max', size=(8, 1))],
                      [sg.Text('Y limits', size=(25, 1)),
                       sg.InputText('min', key='y_min', size=(8, 1)), sg.InputText('max', key='y_max', size=(8, 1))]
                  ])],
        [sg.Frame('Thresholding:',
                  [[sg.Text('Read length threshold value:', size=(25, 1)),
                    sg.InputText('min', key='-min_length_thr-', size=(8, 1)),
                   sg.InputText('max', key='-max_length_thr-', size=(8, 1))],
                   [sg.B('Reload plot'), sg.B('Save values'), sg.B('Cancel')]])],
    ]

    window2 = sg.Window(title='Align score filtering', layout=layout)
    window2.Finalize()

    # \\  -------- PYSIMPLEGUI -------- //
    bins_def = 100
    figsize_def = (8, 4)
    bins = bins_def
    figsize = figsize_def

    # Plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1, height_ratios=[1])

    ax0 = plt.subplot(gs[0])
    ax0.hist(read_length_l, bins=bins, density=True)

    ax0.set_xlabel('Read length')
    ax0.set_ylabel('Frequency')

    plt.tight_layout()

    # ------------------------------- Instead of plt.show()
    draw_figure_w_toolbar(window2['fig_cv'].TKCanvas, fig, window2['controls_cv'].TKCanvas)
    plt.close(fig)
    # \\  -------- PYPLOT -------- //
    while True:
        event, values = window2.Read()

        # ------------------------------- PASTE YOUR MATPLOTLIB CODE HERE
        if event == sg.WIN_CLOSED or event == 'Cancel':
            plt.close(fig)
            window2.Close()
            break

        elif event == 'Reload plot':
            if values['-min_length_thr-'].isdigit() == False or values['-max_length_thr-'].isdigit() == False:
                sel_thr = False

            else:
                sel_thr = True
                Min_length_thr = int(values['-min_length_thr-'])
                Max_length_thr = int(values['-max_length_thr-'])


            if values['bins'].isdigit() == True:
                bins = int(values['bins'])
            else:
                bins = bins_def

            if values['w'].isdigit() == False or values['h'].isdigit() == False:
                figsize = figsize_def
            else:
                figsize = (int(values['w']), int(values['h']))

            norm = int(values['chk_norm'])

            try:
                x_min = float(values['x_min'])
                x_max = float(values['x_max'])
                x_lim = True
            except ValueError:
                x_lim = False

            try:
                y_min = float(values['y_min'])
                y_max = float(values['y_max'])
                y_lim = True

            except ValueError:
                y_lim = False

            y, x = plt.hist(read_length_l, bins=bins)[:2]
            plt.close()

            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 1, height_ratios=[1])
            ax0 = plt.subplot(gs[0])
            ax0.hist(read_length_l, bins=bins, density=norm)

            if sel_thr == True:
                ax0.axvspan(Min_length_thr, Max_length_thr, color=(0, 1.0, 0), alpha=0.5,
                            label=str(Min_length_thr) + ' - ' + str(Max_length_thr))
                ax0.legend(bbox_to_anchor=(1, 1), loc='upper left', title='Alignment score')

            if x_lim == True:
                ax0.set_xlim(x_min, x_max)
            if y_lim == True:
                ax0.set_ylim(y_min, y_max)

            ax0.set_xlabel('Read length')
            if norm == True:
                ax0.set_ylabel('Normalized frequency')
            else:
                ax0.set_ylabel('Frequency')

            plt.tight_layout()
            draw_figure_w_toolbar(window2['fig_cv'].TKCanvas, fig, window2['controls_cv'].TKCanvas)
            plt.close()

        elif event == 'Save values':
            if values['-min_length_thr-'].isdigit() == False or values['-max_length_thr-'].isdigit() == False:
                sg.Popup('Thresholding parameters not provided.')

            else:
                Min_length_thr = int(values['-min_length_thr-'])
                Max_length_thr = int(values['-max_length_thr-'])
                plt.close(fig)
                window2.Close()

                # Filtering data
                data_df_filt = data_df.loc[(Min_length_thr < data_df['Read length']) & (data_df['Read length'] < Max_length_thr)]
                filt_records = data_df_filt['Records'].values.tolist()
                filt_seq = data_df_filt['Sequences'].values.tolist()

                # Exporting thresholding value
                with open(exp_folder + '/Thresholding_info.csv', 'w', newline='') as csvfile:

                    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                            quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['Thresholding score', Min_length_thr, Max_length_thr])

                return filt_records, filt_seq

# Main gui
def ONT_filter_main_gui():
    import PySimpleGUI as sg
    import glob, os
    from Bio import SeqIO

    left_col = [
        [sg.Frame('Data information:',
                  [[sg.Text('Hdf file with extracted data'),
                    sg.In(size=(50, 1), enable_events=True, key='data_df'),
                    sg.FileBrowse()],
                   [sg.Text('Number of reads to export:', size=(25, 1)),
                    sg.InputText(key='read_n_thr', size=(30, 1))]
                   ])],
        [sg.Text('Folder to save data to:'),
         sg.In(size=(50, 1), enable_events=True, key='export_folder'),
         sg.FolderBrowse()],
        [sg.Button('Load data'), sg.Button('Cancel')]
    ]
    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 30), key='OUTPUT')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_read_filter = sg.Window('Read filtering', layout, resizable=True)

    #
    while True:
        event, values = window_read_filter.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_read_filter.close()
            break

        if event == 'Load data':
            window_read_filter['OUTPUT'].update(value='Processing data' + '\n', append=True)
            window_read_filter.refresh()

            try:
            
                data_df = values['data_df']
                if len(data_df) == 0:
                    sg.Popup('Hdf file with extracted sequences not provided.')
                    break

                else:
                    window_read_filter['OUTPUT'].update(value='Data dataframe loaded' + '\n', append=True)
                    window_read_filter['OUTPUT'].update(value=data_df + '\n', append=True)
                    window_read_filter.refresh()

                read_n_thr = values['read_n_thr']
                if len(read_n_thr) == 0:
                    read_filt = False

                else:
                    read_filt = True
                    read_n_thr = int(read_n_thr)

                exp_folder = values['export_folder']
                if len(exp_folder) == 0:
                    sg.Popup('Export folder not specified.')
                    break

                filt_fl_record_l, filt_fl_seq_l = read_filter(df_exp=data_df, exp_folder=exp_folder)

                if read_filt == True:
                    if len(filt_fl_seq_l) > read_n_thr:
                        filt_fl_seq_l = filt_fl_seq_l[:read_n_thr + 1]
                    if len(filt_fl_record_l) > read_n_thr:
                        filt_fl_record_l = filt_fl_record_l[:read_n_thr + 1]

                SeqIO.write(filt_fl_record_l, exp_folder + "//filtered_records.fasta", 'fasta')
                SeqIO.write(filt_fl_seq_l, exp_folder + "//filtered_read_seq.fasta", 'fasta')

                window_read_filter['OUTPUT'].update(value='Results exported' + '\n', append=True)
                window_read_filter.refresh()

            except ValueError:
                sg.Popup('Could not proceed with processing. Check if all parameters are provided correctly.')

ONT_filter_main_gui()