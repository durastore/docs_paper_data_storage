# Consensus generation
def cons_gen(data_file, export_folder, MSA_soft, base_thr):
    import Bio
    from Bio import AlignIO
    from Bio.Align import MultipleSeqAlignment
    from Bio.Align.AlignInfo import SummaryInfo
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.Seq import Seq

    if MSA_soft == 'clustal':
        alignment = AlignIO.read(data_file, MSA_soft)
    if MSA_soft == 'muscle':
        alignment = AlignIO.read(data_file, 'fasta')
    summary = SummaryInfo(alignment)
    consensus = summary.gap_consensus(base_thr, "N")
    print(consensus)
    consensus = consensus.replace("N", "")
    consensus = consensus.replace("-", "")
    print(consensus)
    record = SeqRecord(consensus,
                       'Consensus',
                       'Consensus',
                       'Consensus'
                       )
    SeqIO.write([record], export_folder + '//consensus.fasta',
                "fasta")

# Main gui
def ONT_cons_main_gui():
    import PySimpleGUI as sg
    import csv
    import glob, os
    from Bio import SeqIO

    left_col = [
        [sg.Frame('Data information:',
                  [[sg.Text('.aln file with aligned sequences:'),
                    sg.In(size=(50, 1), enable_events=True, key='data_aln'),
                    sg.FileBrowse()],
                   ])],
        [sg.Frame('MSA program used for alignment:',
                  [[                sg.Combo(['clustal',
                                              'muscle'], default_value='clustal', enable_events=True,
                                             key='ch_msa')]
                   ])],
        [sg.Text('Base threhold:', size=(36, 1)),
         sg.InputText('0.25', key='base_thr', size=(8, 1))],
        [sg.Text('Folder to save consensus sequence to:'),
         sg.In(size=(50, 1), enable_events=True, key='export_folder'),
         sg.FolderBrowse()],
        [sg.Button('Load data'), sg.Button('Cancel')]
    ]
    right_col = [
        [sg.Frame('Output:', [[sg.Multiline("", size=(50, 30), key='OUTPUT')]])]
    ]

    layout = [[sg.Column(left_col, element_justification='c'),
               sg.Column(right_col, element_justification='c', vertical_alignment="top")]]
    window_cons_gen = sg.Window('Consensus generation', layout, resizable=True)

    #
    while True:
        event, values = window_cons_gen.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_cons_gen.close()
            break

        msa_soft = values['ch_msa']
        base_thr = float(values['base_thr'])

        

        if event == 'Load data':
            window_cons_gen['OUTPUT'].update(value='Processing data' + '\n', append=True)
            window_cons_gen.refresh()

            try:

                data_aln_file = values['data_aln']
                if len(data_aln_file) == 0:
                    sg.Popup('aln file with aligned data not provided.')
                    break

                window_cons_gen['OUTPUT'].update(value='Data file loaded:' + '\n', append=True)
                window_cons_gen.refresh()

                exp_folder = values['export_folder']
                if len(exp_folder) == 0:
                    sg.Popup('Export folder not specified.')
                    break

                window_cons_gen['OUTPUT'].update(value='Consensus generation started' + '\n', append=True)
                window_cons_gen.refresh()

                cons_gen(data_file=data_aln_file, export_folder=exp_folder, MSA_soft=msa_soft, base_thr=base_thr)

                window_cons_gen['OUTPUT'].update(value='Consensus generation finished' + '\n', append=True)
                window_cons_gen.refresh()

            except ValueError:
                sg.Popup('Could not proceed with processing. Check if all parameters are provided correctly.')

ONT_cons_main_gui()