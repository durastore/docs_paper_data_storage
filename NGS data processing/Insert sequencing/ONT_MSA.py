#Clustal MSA
def msa (data_file, clustal, muscle):
    import Bio
    import subprocess
    import os

    if clustal == True:
        clustalw_exe = r"C:\ClustalW2\clustalw2.exe"
        cmd = clustalw_exe + " -infile=" + str(data_file)
        results = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)

    if muscle == True:
        muscle_exe = r"C:\MUSCLE\muscle.exe"
        cmd = muscle_exe + " -in " + str(data_file) +" -out " +str(data_file)[:-6] + '-aligned.fasta'
        results = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)

# Main gui
def ONT_MSA_main_gui ():
    import PySimpleGUI as sg
    import csv
    import glob, os
    from Bio import SeqIO

    left_col = [
        [sg.Frame('Data information:',
                  [[sg.Text('Fasta file with extracted sequences:'),
                    sg.In(size=(50, 1), enable_events=True, key='data_fasta'),
                    sg.FileBrowse()]
                   ])],
        [sg.Frame('MSA program:',
                  [[sg.Checkbox('Clustal', size=(30, 1), key='chk_clustal')],
                   [sg.Checkbox('MUSCLE', size=(30, 1), key='chk_muscle')]
                   ])],
        [sg.Button('Load data'), sg.Button('Cancel')]
    ]
    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 30), key='OUTPUT')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_msa = sg.Window('Multiple sequence alignment', layout, resizable=True)
    #
    while True:
        event, values = window_msa.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_msa.close()
            break

        muscle = values['chk_muscle']
        clustal = values['chk_clustal']

        if muscle == False and clustal == False:
            sg.Popup('No MSA program selected.')
            break


        if event == 'Load data':
            window_msa['OUTPUT'].update(value='Processing data' + '\n', append=True)
            window_msa.refresh()

            try:


                data_file = values['data_fasta']
                if len(data_file) == 0:
                    sg.Popup('FASTA file with extracted data not provided.')
                    break

                window_msa['OUTPUT'].update(value='Data file loaded:' + '\n', append=True)
                window_msa.refresh()

                window_msa['OUTPUT'].update(value='MSA started' + '\n', append=True)
                window_msa.refresh()
                
                msa(data_file=data_file, clustal=clustal, muscle=muscle)

                window_msa['OUTPUT'].update(value='MSA finished' + '\n', append=True)
                window_msa.refresh()

            except ValueError:
                sg.Popup('Could not proceed with processing. Check if all parameters are provided correctly.')

ONT_MSA_main_gui ()