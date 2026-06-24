#Functions used:
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

#Function for generating colors for plotting and drawing on images
def color_picker(colormap, channels):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    cmap = plt.cm.get_cmap(name=colormap, lut=channels)
    colors_rgba = []
    colors_bgr = []
    for i in range(channels):
        colors_rgba.append(cmap(i))

    for i in colors_rgba:
        color_bgr = []
        for j in reversed(mpl.colors.to_rgb(i)):
            color_bgr.append(int(j * 255))
        colors_bgr.append(tuple(color_bgr))

    return colors_rgba, colors_bgr

#Functions for exporting cropped images
def loc_to_img_for_det(export_folder, merge_locs_df, img_size, SPR_det, thr_low, thr_high, gauss_size):

    #Import packages
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import copy
    import cv2

    locs_x= merge_locs_df['x'].values.tolist()
    locs_y= merge_locs_df['y'].values.tolist()

    locs_x = [int(i * SPR_det) for i in locs_x]
    locs_y = [int(i * SPR_det) for i in locs_y]

    #Make empty image for detection
    img = np.zeros((int(img_size[0] * SPR_det), int(img_size[1] * SPR_det)))

    #Fill up image with localizations
    for x, y in zip(locs_x, locs_y):
        img[x][y]=img[x][y]+1.0

    #Normalize intensity
    img = 255.0*((img-thr_low)/(thr_high-thr_low))
    img[img < 0.0] = 0.0

    #Apply Gaussian blur
    img_blur = gaussian_filter(img, sigma=gauss_size)
    cv2.imwrite(export_folder + '/Merge-Render-GS.png', img_blur)
    cv2.destroyAllWindows()

    gs_img_name =export_folder + '/Merge-Render-GS.png'

    return gs_img_name, img_blur

# Image filtering and structure detection
def str_filter_det(img_gs, dil_it_n, export_folder):

    #import packages
    import cv2

    #Make list to store contours areas
    contour_area_l = []

    #Read grayscale image
    image = cv2.imread(img_gs, 0)

    # Binary thresholding after gaussian filtering
    ret3, th3 = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bin_img = th3
    cv2.imwrite(export_folder + '/Merge-Render-GF-Otsu.jpg', th3)

    # Dilation to increase
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 , 3))
    dilation = cv2.dilate(th3, kernel, iterations=dil_it_n)
    cv2.imwrite(export_folder + '/Merge-Render-GF-Otsu-DIL.jpg', dilation)

    #Detection of contours in dilated image
    contour_points = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    image = cv2.imread(img_gs)

    for cnt in contour_points:
        contour_area_l.append(cv2.contourArea(cnt))
        # Define ROI coordinates
        (y1, x1), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(image, (int(y1), int(x1)), int(radius), (0, 0, 255), 1)
    cv2.imwrite(export_folder + '/Merge-all-det-contour.jpg', image)

    return contour_points, bin_img

# Function for structure detection by size thresholding
def size_thr(str_contours, bin_img, img_gs, window_size, pop_spread, export_folder, index_render):

    # import packages used
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from scipy import stats
    from scipy.signal import argrelmax
    import cv2
    import random
    import pandas as pd
    import PySimpleGUI as sg
    import numpy as np

    # For checking size distribution in sample
    Area = []
    for cnt in str_contours:
        Area.append(cv2.contourArea(cnt))

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
        [[sg.Text('Contour area threshold values:', size=(25, 1)), sg.InputText('min', key='-min_area_thr-', size=(8, 1)),
         sg.InputText('max', key='-max_area_thr-', size=(8, 1))],
         [sg.Checkbox('Show kde plot', size=(30, 1), key='chk_kde_show'),sg.Text('Color:', size=(25, 1)),
                    sg.Combo(['blue', 'cyan', 'green', 'yellow', 'magenta', 'purple', 'black', 'grey', 'silver'],
                             default_value='grey', enable_events=True,
                             key='drop1')],
         [sg.Checkbox('Show detected size populations', size=(30, 1), key='chk_pop_show')],
         [sg.B('Reload plot'), sg.B('Save values'), sg.B('Cancel')]])],
    ]

    window2 = sg.Window(title='Origami contour size selection',layout=layout)
    window2.Finalize()

    # \\  -------- PYSIMPLEGUI -------- //
    bins_def = 100
    figsize_def=(8, 4)
    bins = bins_def
    figsize=figsize_def

    # Checking visually population to determine size
    img_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)

    #Plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, height_ratios = [1])

    ax0 = plt.subplot(gs[0])
    ax0.hist(Area, bins=bins, density=True)

    ax0.set_xlabel('Contour area [pixel]')
    ax0.set_ylabel('Frequency')

    ax1 = plt.subplot(gs[1])
    ax1.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    ax1.axis("off")
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
            cv2.destroyAllWindows()
            if values['-min_area_thr-'].isdigit() == False or values['-max_area_thr-'].isdigit() == False:
                sel_thr = False

            else:
                sel_thr = True
                Min_thr = int(values['-min_area_thr-'])
                Max_thr = int(values['-max_area_thr-'])

            pop_show = values['chk_pop_show']

            if values['bins'].isdigit() == True:
                bins = int(values['bins'])
            else:
                bins = bins_def

            if values['w'].isdigit() == False or values['h'].isdigit() == False:
                figsize = figsize_def
            else:
                figsize = (int(values['w']), int(values['h']))

            norm = int(values['chk_norm'])
            kde_show = values['chk_kde_show']
            kde_color = values['drop1']

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


            # Checking visually population to determine size
            # Calculate
            y, x = plt.hist(Area, bins=bins)[:2]
            plt.close()

            x = x.tolist()
            density = stats.gaussian_kde(Area)
            y_n = density(x)

            # Finding populations in peak detection in the probability density plot
            img_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
            if pop_show == True:
                Sizes = []
                peaks = argrelmax(y_n)
                for peaklist in peaks:
                    for peak_point in peaklist:
                        Sizes.append(x[peak_point])

                # Checking visually population to determine size
                Colors_plot, Colors = color_picker(colormap='gist_rainbow', channels=len(Sizes))

                for i in range(len(Sizes)):
                    if i == 0:
                        Min_thr = int(Sizes[i] - (Sizes[i] * pop_spread))
                    else:
                        Min_thr = int(Sizes[i - 1] + (Sizes[i - 1] * pop_spread))
                    Max_thr = int(Sizes[i] + (Sizes[i] * pop_spread))

                    for cnt in str_contours:
                        if cv2.contourArea(cnt) > Min_thr and cv2.contourArea(cnt) < Max_thr:
                            x_cnt, y_cnt, w, h = cv2.boundingRect(cnt)
                            cv2.rectangle(img_rgb, (x_cnt, y_cnt), (x_cnt + w, y_cnt + h), Colors[i], 2)

            if sel_thr == True :
                for cnt in str_contours:
                    if cv2.contourArea(cnt) > Min_thr and cv2.contourArea(cnt) < Max_thr:
                        x_cnt, y_cnt, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(img_rgb, (x_cnt, y_cnt), (x_cnt + w, y_cnt + h), (0, 255, 0), 2)

            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, height_ratios = [1])
            ax0 = plt.subplot(gs[0])
            ax0.hist(Area, bins=bins, density=norm)

            if kde_show == True:
                ax0.plot(x, y_n, color=kde_color)

            if sel_thr == True:
                ax0.axvspan(Min_thr, Max_thr, color=(0, 1.0, 0), alpha=0.5, label=str(Min_thr) + ' - ' + str(Max_thr))
                ax0.legend(bbox_to_anchor=(1, 1), loc='upper left', title='Contour Area [pixel]')
            if pop_show == True:
                for i in range(len(Sizes)):
                    if i == 0:
                        Min_thr = int(Sizes[i] - (Sizes[i] * pop_spread))
                    else:
                        Min_thr = int(Sizes[i - 1] + (Sizes[i - 1] * pop_spread))
                    Max_thr = int(Sizes[i] + (Sizes[i] * pop_spread))
                    ax0.axvspan(Min_thr, Max_thr, color=Colors_plot[i], alpha=0.5,
                                label=str(Min_thr) + ' - ' + str(Max_thr))
                ax0.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Contour Area [pixel]')

            if x_lim == True:
                ax0.set_xlim(x_min, x_max)
            if y_lim == True:
                ax0.set_ylim(y_min, y_max)

            ax0.set_xlabel('Contour area [pixel]')
            if norm == True:
                ax0.set_ylabel('Normalized frequency')
            else:
                ax0.set_ylabel('Frequency')

            ax1 = plt.subplot(gs[1])
            ax1.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
            ax1.axis("off")
            plt.tight_layout()
            draw_figure_w_toolbar(window2['fig_cv'].TKCanvas, fig, window2['controls_cv'].TKCanvas)
            plt.close()
            cv2.destroyAllWindows()

        elif event == 'Save values':
            if values['-min_area_thr-'].isdigit() == False or values['-max_area_thr-'].isdigit() == False:
                sg.Popup('Thresholding parameters not provided.')

            else:
                Min_thr = int(values['-min_area_thr-'])
                Max_thr = int(values['-max_area_thr-'])
                plt.close(fig)
                window2.Close()
                cv2.destroyAllWindows()

                #Filtering contours
                cnt_df = pd.DataFrame(list(
                    zip(str_contours, Area)),
                    columns=['Cnt_points', 'Cnt_area'])

                cnt_df_filt = cnt_df.loc[(Min_thr < cnt_df['Cnt_area']) & (Max_thr > cnt_df['Cnt_area']) ]
                filt_cnt_points = cnt_df_filt['Cnt_points'].values.tolist()

                image = cv2.imread(img_gs)
                index = 0
                for i in range(len(filt_cnt_points)):

                    cnt = filt_cnt_points[i]
                    # Define ROI coordinates
                    (x1, y1), radius = cv2.minEnclosingCircle(cnt)
                    cv2.rectangle(image, (int(x1 - (float(window_size)/2)), int(y1 - (float(window_size)/2))), (int(x1 + (float(window_size)/2)), int(y1 + (float(window_size)/2))), (0, 255, 255), 2)
                    if index_render == True:
                        cv2.putText(image, str(i+1),
                                    (int(x1 + (float(window_size)/2)), int(y1 + (float(window_size)/2) + 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 255),
                                    1)
                cv2.imwrite(export_folder + '/Origami_ROI.jpg', image)

                return filt_cnt_points, Min_thr, Max_thr

#Function for collecting localizations falling into a certain ROI (optional export of images of ROI)
def str_ROI_loc(str_contour_coor, bin_img, locs_df, merge_locs_df, SPR_det, window_size, export_folder, channel_name, str_cnt):
    #import packages used:
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    #calculate ROI window coordinates from contour coordinates
    (y1, x1), radius = cv2.minEnclosingCircle(str_contour_coor)
    e = window_size

    x_min = ((x1 - (e / 2)) / SPR_det)
    x_max = ((x1 + (e / 2)) / SPR_det)
    y_min = ((y1 - (e / 2)) / SPR_det)
    y_max = ((y1 + (e / 2)) / SPR_det)
    w_c_x = (x_max + x_min) / 2
    w_c_y = (y_max + y_min) / 2
    str_c = [x1/ SPR_det, y1/ SPR_det]


    #Extract localization parameters for localizations that fall within the calculated ROI window
    loc_xy_merge_crop = merge_locs_df.loc[(x_min < merge_locs_df['x']) & (x_max > merge_locs_df['x']) & (y_min < merge_locs_df['y']) & (y_max > merge_locs_df['y'])]
    loc_xy_crop = locs_df.loc[(x_min < locs_df['x']) & (x_max > locs_df['x']) & (y_min < locs_df['y']) & (y_max > locs_df['y'])]
    x = loc_xy_crop['x'].values.tolist()
    y = loc_xy_crop['y'].values.tolist()
    x_merge = loc_xy_merge_crop['x'].values.tolist()
    y_merge = loc_xy_merge_crop['y'].values.tolist()

    x_c = (max(x_merge) + min(x_merge)) / 2
    y_c = (max(y_merge) + min(y_merge)) / 2

    x_diff = w_c_x - x_c
    y_diff = w_c_y - y_c

    for i in range(len(x)):
        x[i] = x[i] - x_min + x_diff
        y[i] = y[i] - y_min + y_diff

    #Create arrays with values for exporting
    str_coor = np.asarray(list(zip(x,y)))
    str_trace = np.asarray(list(zip(np.asarray(loc_xy_crop['frame'].values.tolist()),np.asarray(loc_xy_crop['photons'].values.tolist()))))
    str_loc_prec = np.asarray(list(zip(np.asarray(loc_xy_crop['lpx'].values.tolist()),np.asarray(loc_xy_crop['lpy'].values.tolist()))))

    #For exporting the binary image of the ROI
    ROI_y1, ROI_y2, ROI_x1, ROI_x2 = int(x_min * SPR_det), int(x_max * SPR_det), int(y_min * SPR_det), int(y_max * SPR_det)
    
    #Preparing data to export:
    exp_data_values = []
    exp_data_name = []
    
    exp_data_values.append(str_coor)
    exp_data_name.append('loc_coor')
    
    exp_data_values.append(str_trace)
    exp_data_name.append('loc_trace')
    
    exp_data_values.append(str_loc_prec)
    exp_data_name.append('loc_prec')



    df_str = pd.DataFrame(list(zip(*exp_data_values)), columns=exp_data_name)

    return str_c, df_str, exp_data_name

def PAINT_segment_main_gui():
    # import modules used
    import PySimpleGUI as sg
    import csv

    # Input information needed for processing

    left_col = [
        [sg.Frame('Data information:',
                  [[sg.Text('Number of channels in PAINT data:', size=(36, 1)),
                    sg.InputText(key='PAINT_channel_n', size=(8, 1))],
                   [sg.Text('Csv file with file size:'),
                    sg.In(size=(50, 1), enable_events=True, key='img_size_csv'),
                    sg.FileBrowse()]
                   ])],
        [sg.Frame('Image processing parameters:',
                  [[sg.Text('Sub-sampling:', size=(36, 1)), sg.InputText('20', key='SPR', size=(8, 1))],
                   [sg.Text('Crop window size:', size=(36, 1)), sg.InputText('60', key='Window_size', size=(8, 1))],
                   [sg.Text('Normalization intensity:', size=(36, 1)),
                    sg.InputText('0.2', key='min_int_thr', size=(8, 1)),
                    sg.InputText('1.5', key='max_int_thr', size=(8, 1))],
                   [sg.Text('Gaussian blur kernel size:', size=(36, 1)),
                    sg.InputText('2.0', key='gauss_kernel', size=(8, 1))],
                   [sg.Text('Number of dilations used for origami detection:', size=(36, 1)),
                    sg.InputText('10', key='dil_n', size=(8, 1))],
                   ])],
        [sg.Button('Select localization files'), sg.Button('Cancel')]
    ]
    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 10), key='OUTPUT')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_render_load = sg.Window('PAINT data segmenting main', layout, resizable=True)

    #
    while True:
        event, values = window_render_load.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_render_load.close()
            break

        if event == 'Select localization files':
            try:
                PAINT_channel_n = int(values['PAINT_channel_n'])
                thr_min = float(values['min_int_thr'])
                thr_max = float(values['max_int_thr'])
                gauss_kernel = float(values['gauss_kernel'])
                SPR = int(values['SPR'])
                dil_n = int(values['dil_n'])
                win_size = int(values['Window_size'])

                img_size_csv_file = values['img_size_csv']
                with open(img_size_csv_file, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    img_size = [ row for row in spamreader][4]
                    img_size = [int(value) for value in img_size]
                rendering_param = [thr_min, thr_max, gauss_kernel, SPR, dil_n, win_size, img_size]
                PAINT_segment_gui(channel_n=PAINT_channel_n, rendering_param=rendering_param)

            except ValueError:
                sg.Popup('Could not proceed with rendering. Check if all parameters are provided correctly.')

def PAINT_segment_gui(channel_n, rendering_param):
    # import modules used
    import pandas as pd
    import PySimpleGUI as sg
    import csv
    from tqdm import tqdm

    #Define imported rendering parameters
    thr_min, thr_max, gauss_kernel, SPR, dil_n, win_size, img_size = rendering_param
    # Input information needed for processing
    #
    left_col = []
    for i in range(channel_n):
        left_col.append([sg.Frame('Channel ' + str(i) + ' information:',
                                  [[sg.Text('PAINT localization file for channel:'),
                                    sg.In(size=(50, 1), enable_events=True, key='file-ch-' + str(i)),
                                    sg.FileBrowse()],
                                   [sg.Text('Name of imager sequence used for channel:', size=(25, 1)),
                                    sg.InputText(key='PAINT_img_name-ch-' + str(i), size=(8, 1))]
                                   ])])
    left_col.append([sg.Frame('Exported data information:',
                                  [[sg.Text('Folder to save data to:'),
                                    sg.In(size=(50, 1), enable_events=True, key='export_folder'),
                                    sg.FolderBrowse()],
                                   [sg.Checkbox('Add origami ROI indeces to rendered image', size=(36, 1),
                                                key='chk_index_render')],
                                   [sg.Text('Number of probes to process:', size=(36, 1)),
                                    sg.InputText('', key='probe_n_limit', size=(8, 1))],
                                   ])])
    left_col.append([sg.Button('Segment data'), sg.Button('Cancel')])

    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 10), key='OUTPUT')]])],
        [sg.Frame('Progress:', [[sg.Multiline("", size=(50, 1), key='OUTPUT2')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_data_segment = sg.Window('PAINT data segmenting', layout, resizable=True)

    #
    loc_files = []
    click_memory = []
    for i in range(channel_n):
        click_memory.append(0)

    while True:
        event, values = window_data_segment.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_data_segment.close()
            break

        #Create lists to store data for processing
        channel_names = []
        data_set_locs = []

        #Create lists for exporting data
        origami_index = []
        origami_coor = []
        segmented_locs = []

        if event == 'export_folder':
            exp_folder = values['export_folder']

        for j in range(channel_n):

            if event == 'file-ch-' + str(j):

                click_memory[j] = click_memory[j] + 1
                if click_memory[j] < 2:
                    loc_files.append(values['file-ch-' + str(j)])
                else:
                    loc_files[j] = values['file-ch-' + str(j)]

        if event == 'Segment data':
            window_data_segment['OUTPUT'].update(
                value='')
            window_data_segment.refresh()

            if len(exp_folder) == 0 :
                sg.Popup('Folder for exporting data not specified.')
                break

            else:
                if '' in loc_files:
                    sg.Popup('Dataset(s) not specified.')
                    break

                else:
                    for h in range(len(loc_files)):
                        channel_names.append(str(values['PAINT_img_name-ch-' + str(h)]))

                    if '' in channel_names:
                        sg.Popup('Channel name(s) not provided.')
                        break

                    else:
                        # exp_loc_coor = values['chk_exp_loc_coor']
                        # exp_loc_trace = values['chk_exp_loc_trace']
                        # exp_loc_prec = values['chk_exp_loc_prec']
                        # exp_img = values['chk_exp_img']
                        # exp_loc_plots = values['chk_exp_loc_plot']
                        index_render = values['chk_index_render']
                        # data_format = values['drop1']

                        if values['probe_n_limit'].isdigit() == False:
                            probe_proc_limit = False
                        else:
                            probe_proc_limit = True
                            probe_n_limit = int(values['probe_n_limit'])

                        for k in range(channel_n):
                            loc_file = loc_files[k]
                            loc_file_ch_name = channel_names[k]

                            window_data_segment['OUTPUT'].update(value='Localization file loaded:' + '\n', append=True)
                            window_data_segment.refresh()
                            window_data_segment['OUTPUT'].update(value=loc_file + '\n', append=True)
                            window_data_segment.refresh()
                            window_data_segment['OUTPUT'].update(value=loc_file_ch_name + '\n', append=True)
                            window_data_segment.refresh()

                            # Import loc file and transform it into a np array
                            locs_df = pd.read_hdf(loc_file, key='locs')
                            data_set_locs.append(locs_df)
                            segmented_locs.append([])

                        #Determining origami coordinates
                        data_set_merge = pd.concat(data_set_locs)

                        # Reconstruct image from localizations
                        window_data_segment['OUTPUT'].update(value='Rendering localizations and exporting super-resolved images' + '\n', append=True)
                        window_data_segment.refresh()
                        gs_img_name, gs_img = loc_to_img_for_det(export_folder=exp_folder, merge_locs_df=data_set_merge,
                                                                 img_size=img_size, SPR_det=SPR, thr_low=thr_min,
                                                                 thr_high=thr_max, gauss_size=gauss_kernel)

                        # Detecting contours in reconstructed image
                        window_data_segment['OUTPUT'].update(value='Detecting contours in super-resolved image' + '\n',
                                                append=True)
                        window_data_segment.refresh()
                        contour_points, bin_img = str_filter_det(img_gs=gs_img_name, dil_it_n=dil_n, export_folder=exp_folder)

                        # Filter noise out from contours
                        window_data_segment['OUTPUT'].update(
                            value='Determining origami ROI coordinates by picking size interval for origami contours' + '\n',
                            append=True)
                        window_data_segment.refresh()
                        filt_cnt_points, Min_thr, Max_thr = size_thr(str_contours=contour_points, bin_img=bin_img,img_gs=gs_img_name,
                                                                     pop_spread=0.6,export_folder=exp_folder, window_size=win_size,index_render=index_render)

                        ##Cropping structures out
                        window_data_segment['OUTPUT'].update(
                            value='Extracting localizations clustered into spots in origami ROIs' + '\n',
                            append=True)
                        window_data_segment.refresh()
                        str_cnt = 0
                        if probe_proc_limit == False:
                            it_n = len(filt_cnt_points)
                        else:
                            it_n = probe_n_limit
                        with tqdm(total=it_n, ncols=50) as pbar:

                            for l in range(it_n):
                                cnt = filt_cnt_points[l]

                                pbar.update(1)
                                window_data_segment['OUTPUT2'].update(value=pbar)
                                window_data_segment.refresh()
                                str_cnt = str_cnt + 1
                                origami_index.append(str_cnt)
                                # Extract localizations falling within origami

                                for h in range(len(data_set_locs)):

                                    channel_name = channel_names[h]
                                    locs_df = data_set_locs[h]

                                    str_coor, df_str, exp_data_header = str_ROI_loc(
                                        str_contour_coor=cnt, bin_img=gs_img, locs_df=locs_df, merge_locs_df=data_set_merge, SPR_det=SPR,
                                        window_size=win_size, export_folder=exp_folder, channel_name=channel_name, str_cnt=str_cnt)

                                    segmented_locs[h].append(df_str)
                                origami_coor.append(str_coor)

                        #Exporting data to hdml5 file
                        with open(exp_folder + '/Segment_info.csv', 'w', newline='') as csvfile:

                            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"',
                                                    quoting=csv.QUOTE_MINIMAL)
                            spamwriter.writerow(['Number of channels', channel_n])
                            spamwriter.writerow(['Data path', 'Channel name'])
                            for k in range(channel_n):
                                data_path = loc_files[k]
                                channel_name = channel_names[k]
                                spamwriter.writerow([data_path, channel_name])
                                
                            spamwriter.writerow(['Thresholding parameters'])
                            spamwriter.writerow(['Sub pixelation rate used for rendering', SPR])
                            spamwriter.writerow(['Kernel size for gaussian bluring used', gauss_kernel])
                            spamwriter.writerow(['Number of dilation cycles utilized', dil_n])
                            spamwriter.writerow(['Contour area range selected', Min_thr, Max_thr])
                            spamwriter.writerow(['ROI window size used', win_size])
                            spamwriter.writerow(['Types of exported localization values'])
                            spamwriter.writerow(['loc_coor','loc_trace','loc_prec'])

                        exported_data = [origami_index]+[origami_coor]+segmented_locs
                        df_exp = pd.DataFrame(list(zip(*exported_data)),
                                              columns=['Str_index', 'Str_pos']+channel_names)

                        df_exp.to_hdf(exp_folder + '/Segmented_loc_data.h5', key='seg_locs', mode='w')
                        window_data_segment['OUTPUT'].update(
                            value='Data exported' + '\n',
                            append=True)
                        window_data_segment.refresh()

