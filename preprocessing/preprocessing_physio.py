import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import mne
import argparse, random

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--map_size_gb', type=int, default=6)
    p.add_argument('--seed', type=int, default=0)

    p.add_argument('--sfreq', type=float, default=200.0)
    p.add_argument('--l_freq', type=float, default=0.3)
    p.add_argument('--notch', type=float, default=60.0)
    p.add_argument('--win_sec', type=float, default=4.0)
    p.add_argument('--take_last_n', type=int, default=800)
    p.add_argument('--batch_commit', type=int, default=512)

    # сплиты: либо проценты, либо точные числа
    p.add_argument('--p_train', type=float, default=None)
    p.add_argument('--p_val', type=float, default=None)
    p.add_argument('--p_test', type=float, default=None)
    p.add_argument('--n_train', type=int, default=None)
    p.add_argument('--n_val', type=int, default=None)
    p.add_argument('--n_test', type=int, default=None)
    return p.parse_args()

def make_subject_list(root_dir, tasks):
    # субъект = директория первого уровня, где есть хотя бы один нужный EDF
    subjects = []
    for name in sorted(os.listdir(root_dir)):
        subj_dir = os.path.join(root_dir, name)
        if not os.path.isdir(subj_dir):
            continue
        has_any = any(os.path.exists(os.path.join(subj_dir, f'{name}R{t}.edf')) for t in tasks)
        if has_any:
            subjects.append(name)
    return subjects

def split_subjects(subjects, seed, p=None, n=None):
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)

    if p is not None:
        p_train, p_val, p_test = p
        total = len(shuffled)
        n_train = int(round(total * p_train))
        n_val   = int(round(total * p_val))
        n_test  = total - n_train - n_val
    else:
        n_train, n_val, n_test = n
        assert n_train + n_val + n_test <= len(shuffled), "Размеры сплитов больше числа субъектов"

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train+n_val]
    test  = shuffled[n_train+n_val:n_train+n_val+n_test]
    return {'train': train, 'val': val, 'test': test}


def run():
    args = parse_args()

    tasks = ['04', '06', '08', '10', '12', '14']  # как в исходнике
    selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                         'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                         'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                         'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                         'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                         'O1..', 'Oz..', 'O2..', 'Iz..']

    subjects = make_subject_list(args.root_dir, tasks)
    assert len(subjects) > 0, "Не найдены валидные субъекты в root_dir"

    # выбор способа задания сплитов
    if args.p_train is not None:
        splits_by_subject = split_subjects(subjects, args.seed, p=(args.p_train, args.p_val, args.p_test))
    else:
        splits_by_subject = split_subjects(subjects, args.seed, n=(args.n_train, args.n_val, args.n_test))

    os.makedirs(args.out_dir, exist_ok=True)
    env = lmdb.open(args.out_dir, map_size=args.map_size_gb * (1024 ** 3))

    index = {'train': [], 'val': [], 'test': []}
    put_counter = 0
    txn = env.begin(write=True)

    for split_name, subj_list in splits_by_subject.items():
        for subj in subj_list:
            subj_dir = os.path.join(args.root_dir, subj)
            for task in tasks:
                edf = os.path.join(subj_dir, f'{subj}R{task}.edf')
                if not os.path.exists(edf):
                    continue
                try:
                    raw = mne.io.read_raw_edf(edf, preload=True)
                except Exception as e:
                    print(f'[WARN] skip {edf}: {e}')
                    continue

                raw.pick_channels(selected_channels, ordered=True)
                if len(raw.info.get('bads', [])) > 0:
                    raw.interpolate_bads()
                raw.set_eeg_reference(ref_channels='average')
                raw.filter(l_freq=args.l_freq, h_freq=None)
                raw.notch_filter((args.notch,))
                raw.resample(args.sfreq)

                events_from_annot, event_dict = mne.events_from_annotations(raw)
                epochs = mne.Epochs(
                    raw, events_from_annot, event_dict,
                    tmin=0, tmax=args.win_sec - 1.0 / raw.info['sfreq'],
                    baseline=None, preload=True
                )
                data = epochs.get_data(units='uV')
                events = epochs.events[:, 2]

                if args.take_last_n is not None and data.shape[-1] >= args.take_last_n:
                    data = data[:, :, -args.take_last_n:]

                bz, ch_nums, _ = data.shape
                data = data.reshape(bz, ch_nums, int(args.win_sec), int(args.sfreq))

                for i, (sample, event) in enumerate(zip(data, events)):
                    if event == 1:
                        continue
                    # сохранить точное поведение исходника (label-сдвиг для 04/08/12)
                    label = (event - 2) if task in ['04', '08', '12'] else event

                    sample_key = f'{subj}R{task}-{i}'
                    pair = {'sample': sample, 'label': label}

                    txn.put(sample_key.encode(), pickle.dumps(pair))
                    index[split_name].append(sample_key)

                    put_counter += 1
                    if (put_counter % args.batch_commit) == 0:
                        txn.commit()
                        txn = env.begin(write=True)

    # сохранить индекс
    txn.put(b'__keys__', pickle.dumps(index))
    txn.commit()
    env.close()

if __name__ == '__main__':
    run()
#
# tasks = ['04', '06', '08', '10', '12', '14'] # select the data for motor imagery
#
# root_dir = '/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/files'
# files = [file for file in os.listdir(root_dir)]
# files = sorted(files)
#
# files_dict = {
#     'train': files[:70],
#     'val': files[70:89],
#     'test': files[89:109],
# }
#
# print(files_dict)
#
# dataset = {
#     'train': list(),
#     'val': list(),
#     'test': list(),
# }
#
#
#
# selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
#                      'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
#                      'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
#                      'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
#                      'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
#                      'O1..', 'Oz..', 'O2..', 'Iz..']
#
# db = lmdb.open('/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/processed_average', map_size=4614542346)
#
# for files_key in files_dict.keys():
#     for file in files_dict[files_key]:
#         for task in tasks:
#             raw = mne.io.read_raw_edf(os.path.join(root_dir, file, f'{file}R{task}.edf'), preload=True)
#             raw.pick_channels(selected_channels, ordered=True)
#             if len(raw.info['bads']) > 0:
#                 print('interpolate_bads')
#                 raw.interpolate_bads()
#             raw.set_eeg_reference(ref_channels='average')
#             raw.filter(l_freq=0.3, h_freq=None)
#             raw.notch_filter((60))
#             raw.resample(200)
#             events_from_annot, event_dict = mne.events_from_annotations(raw)
#             epochs = mne.Epochs(raw,
#                                 events_from_annot,
#                                 event_dict,
#                                 tmin=0,
#                                 tmax=4. - 1.0 / raw.info['sfreq'],
#                                 baseline=None,
#                                 preload=True)
#             data = epochs.get_data(units='uV')
#             events = epochs.events[:, 2]
#             print(data.shape, events)
#             data = data[:, :, -800:]
#             bz, ch_nums, _ = data.shape
#             data = data.reshape(bz, ch_nums, 4, 200)
#             print(data.shape)
#             for i, (sample, event) in enumerate(zip(data, events)):
#                 if event != 1:
#                     sample_key = f'{file}R{task}-{i}'
#                     data_dict = {
#                         'sample': sample, 'label': event - 2 if task in ['04', '08', '12'] else event
#                     }
#                     txn = db.begin(write=True)
#                     txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
#                     txn.commit()
#                     dataset[files_key].append(sample_key)
#
# txn = db.begin(write=True)
# txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
# txn.commit()
# db.close()
