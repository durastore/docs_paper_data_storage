#Function to filter sequences based on the inserts flanking sequences and generation of forward seq
def read_extract(read_file_l, fl_seq_l, ref_seq, exp_folder, window):
    from tqdm import tqdm
    import pandas as pd
    from Bio import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO
    from Bio import pairwise2

    FW_seq, REV_seq = fl_seq_l
    rc_FW_seq = Seq.reverse_complement(FW_seq)
    rc_REV_seq = Seq.reverse_complement(REV_seq)

    fw_insert_seq = [FW_seq, REV_seq]
    rev_insert_seq = [rc_REV_seq, rc_FW_seq]

    file_n = len(read_file_l)

    read_len_l = []
    insert_len_l = []
    record_l = []
    seq_l = []

    with tqdm(total=file_n, ncols=50) as pbar:
        for file in read_file_l:
            pbar.update(1)
            window['OUTPUT'].update(value=pbar)
            window.refresh()
            try:
                for record in SeqIO.parse(file, "fastq"):
    
                    read_seq = record.seq
                    read_id = record.id
                    read_name = record.name
                    read_description = record.description
                    if fw_insert_seq[0] in read_seq and fw_insert_seq[1] in read_seq:
                    #if fw_insert_seq[0] and fw_insert_seq[1] in read_seq:
                        start = read_seq.find(fw_insert_seq[0])
                        end = read_seq.find(fw_insert_seq[1])
                        insert_seq = read_seq[start:end + len(fw_insert_seq[1])]
                        if len(insert_seq) != 0:
                            read_len_l.append(len(read_seq))
                            insert_len_l.append(len(insert_seq))
                            record_l.append(record)
                            new_record = SeqRecord(
                                            insert_seq,
                                            read_id,
                                            read_name,
                                            read_description
                                        )
                            seq_l.append(new_record)
    
                    #else:
                        #if rev_insert_seq[0] and rev_insert_seq[1] in read_seq:
                    elif rev_insert_seq[0] in read_seq and rev_insert_seq[1] in read_seq:
                        start = read_seq.find(rev_insert_seq[0])
                        end = read_seq.find(rev_insert_seq[1])
                        insert_seq = Seq.reverse_complement(read_seq[start:end + len(rev_insert_seq[1])])
                        if len(insert_seq) != 0:
                            read_len_l.append(len(read_seq))
                            insert_len_l.append(len(insert_seq))
                            record_l.append(record)
                            new_record = SeqRecord(
                                insert_seq,
                                read_id,
                                read_name,
                                read_description
                            )
                            seq_l.append(new_record)
            except:
                pass
    df_exp = pd.DataFrame(list(zip(read_len_l, insert_len_l, record_l, seq_l)),
                          columns=['Read length', 'Insert_l', 'Records', 'Sequences'])

    df_exp.to_hdf(exp_folder + '//Extracted_reads.h5', key='read_l', mode='w')

#Main gui
def ONT_extract_main_gui ():
    import PySimpleGUI as sg
    import glob, os
    from Bio import SeqIO

    left_col = [
        [sg.Frame('Data information:',
                  [[sg.Text('Folder with fastq files:'),
                    sg.In(size=(50, 1), enable_events=True, key='read_folder'),
                    sg.FolderBrowse()],
                   [sg.Text('Fasta file with reference sequence:'),
                    sg.In(size=(50, 1), enable_events=True, key='ref_seq_fasta'),
                    sg.FileBrowse()]
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
    window_read_extract = sg.Window('Read extraction', layout, resizable=True)

    #
    while True:
        event, values = window_read_extract.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            window_read_extract.close()
            break

        if event == 'Load data':
            window_read_extract['OUTPUT'].update(value='Processing data' + '\n', append=True)
            window_read_extract.refresh()

            try:
                fastq_folder = values['read_folder']
                if len(fastq_folder) == 0:
                    sg.Popup('Folder with reads not provided.')
                    break
                
                os.chdir(fastq_folder)
                files_list = []
                for file in glob.glob("*.fastq"):
                    files_list.append(file)
                
                if len(files_list) == 0:
                    sg.Popup('No fast files found in folder.')
                    break

                window_read_extract['OUTPUT'].update(value='Fastq files loaded:' + '\n', append=True)
                window_read_extract.refresh()

                ref_seq_file = values['ref_seq_fasta']
                if len(ref_seq_file) == 0:
                    sg.Popup('Fasta file with reference sequence not provided.')
                    break

                else:
                    ref_seq = [seq_record for seq_record in SeqIO.parse(ref_seq_file, "fasta")][0].seq
                    window_read_extract['OUTPUT'].update(value='Reference sequence loaded:' + '\n', append=True)
                    window_read_extract['OUTPUT'].update(value=ref_seq_file + '\n', append=True)
                    window_read_extract.refresh()

                FW_seq = ref_seq[:8]
                REV_seq = ref_seq[-8:]

                exp_folder = values['export_folder']
                if len(exp_folder) == 0:
                    sg.Popup('Export folder not specified.')
                    break

                read_extract(read_file_l=files_list, fl_seq_l=[FW_seq, REV_seq], ref_seq=ref_seq, exp_folder=exp_folder, window=window_read_extract)
                
                window_read_extract['OUTPUT'].update(value='Results exported' + '\n', append=True)
                window_read_extract.refresh()
                
            except ValueError:
                sg.Popup('Could not proceed with processing. Check if all parameters are provided correctly.')
                
ONT_extract_main_gui ()