def ONT_ref_seq_main_gui():
    import PySimpleGUI as sg
    import csv
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq
    
    left_col = [
        [sg.Frame('Input information:',
                  [[sg.Text('Reference sequence:', size=(20, 1)),
                    sg.InputText(key='ref_seq', size=(80, 1))],
                   [sg.Text('Reference sequence name:', size=(20, 1)),
                    sg.InputText(key='ref_seq_name', size=(80, 1))]
                   ])],
        [sg.Text('Folder to save data to:'),
         sg.In(size=(50, 1), enable_events=True, key='export_folder'),
         sg.FolderBrowse()],
        [sg.Button('Save'), sg.Button('Cancel')]
    ]
    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 30), key='OUTPUT')]])]
    ]
    
    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_ref_exp = sg.Window('Reference sequence export', layout, resizable=True)
    
    #
    while True:
        event, values = window_ref_exp.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_ref_exp.close()
            break
    
        if event == 'Save':
            window_ref_exp['OUTPUT'].update(value='Exporting sequence' + '\n', append=True)
            window_ref_exp.refresh()
    
            try:
                exp_folder = values['export_folder']
                if len(exp_folder) == 0:
                    sg.Popup('Export folder not specified.')
                    break

                ref_seq = values['ref_seq']
                if len(ref_seq) == 0:
                    sg.Popup('Reference sequence not provided.')
                    break

                ref_seq_name = values['ref_seq_name']
                if len(ref_seq_name) == 0:
                    sg.Popup('Reference sequence name not provided.')
                    break
                    
                else:

                    ref_seq = Seq(ref_seq)
                    record = SeqRecord(ref_seq,
                                       ref_seq_name,
                                       ref_seq_name,
                                       ref_seq_name
                                       )
                    SeqIO.write([record], exp_folder + '//'+str(ref_seq_name)+'.fasta',
                                "fasta")

                window_ref_exp['OUTPUT'].update(value='Reference sequence exported' + '\n', append=True)
                window_ref_exp.refresh()
    
            except ValueError:
                sg.Popup('Could not proceed with rendering. Check if all parameters are provided correctly.')
            