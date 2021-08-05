import os
from subprocess import call
from glob2 import glob
import nibabel as nib
from shutil import copy2, copytree
import json
from threading import Thread
from sklearn.model_selection import KFold


def prepare_data():

    output_path = 'data/kfold'
    xfer_start = './tmp/ckpt_xfer_210305/checkpoint-8212*'
    checkpoint_file = 'tmp/ckpt_201102/checkpoint-latest'
    epochs = 200
    iters = 3000
    batch_size = 6

    # Configuration file
    base_confile = 'config_MH_xfer.json'
    confile = 'conf_kfold_sim_real_{:02d}.json'

    # Load base
    with open(base_confile, 'r') as f:
        base_df = json.load(f)

    # Get all data files
    base_im_path = 'data/Real_xfer_210305'
    base_im_path_sim = '/home/matt/Documents/Projects/vnet_lung_tumors/data/training'
    im_files = glob(os.path.join(base_im_path, '**', 'image.nii'))
    im_files_sims = glob(os.path.join(base_im_path_sim, '**', 'image.nii'))[:60]

    im_tmp = 'image.nii'
    lab_tmp = 'label.nii.gz'
    lab_tmp_save = lab_tmp.strip('.gz')
    msk_tmp = 'mask.nii.gz'
    msk_tmp_save = msk_tmp.strip('.gz')
    txt_tmp = 'image_type.txt'

    # set up Kfold object
    kf = KFold(n_splits=5, shuffle=False)

    for n, (train_ind, test_ind) in enumerate(kf.split(im_files)):
        print(f'Working on split {n:d}')
        # Set up output path
        spath_train = os.path.join(output_path, 'train_sim_real', f'{n+1:02d}')
        spath_test = os.path.join(output_path, 'test_sim_real', f'{n+1:02d}')
        if not os.path.exists(spath_train):
            os.makedirs(spath_train)
        if not os.path.exists(spath_test):
            os.makedirs(spath_test)

        # Create training configuration file
        df = base_df.copy()
        df['TrainingSetting']['Data']['TrainingDataDirectory'] = os.path.join('./', spath_train)
        df['TrainingSetting']['Data']['TestingDataDirectory'] = os.path.join('./', spath_test)
        df['TrainingSetting']['MaxIterations'] = iters
        df['TrainingSetting']['Epoches'] = epochs
        df['TrainingSetting']['BatchSize'] = batch_size

        df['TrainingSetting']['Restore'] = False
        df['TrainingSetting']['Data']['TestingDataDirectory'] = os.path.join('./', spath_test)
        df['TrainingSetting']['LogDir'] = os.path.join('./', f'tmp/log_kfold_sim_real/{n+1:02d}')
        df['TrainingSetting']['CheckpointDir'] = os.path.join('./', f'tmp/ckpt_kfold_sim_real/{n+1:02d}')
        df['EvaluationSetting']['Data']['EvaluateDataDirectory'] = os.path.join('./', spath_test)

        file = confile.format(n)
        with open(file, 'w') as f:
            json_str = json.dumps(df, indent=4)
            f.write(json_str)

        # # Move training data
        # for nn, ind in enumerate(train_ind):
        #     # Get original files
        #     im_file = im_files[ind]
        #     bpath = os.path.split(im_file)[0]
        #     lab_file = os.path.join(bpath, lab_tmp)
        #     msk_file = os.path.join(bpath, msk_tmp)
        #
        #     # Create output files
        #     spath_inner = os.path.join(spath_train, f'{nn:02d}')
        #     if not os.path.exists(spath_inner):
        #         os.mkdir(spath_inner)
        #     im_file_out = os.path.join(spath_inner, im_tmp)
        #     lab_file_out = os.path.join(spath_inner, lab_tmp_save)
        #     msk_file_out = os.path.join(spath_inner, msk_tmp_save)
        #     txt_file_out = os.path.join(spath_inner, txt_tmp)
        #     with open(txt_file_out, 'w') as f:
        #         f.write(f'Real data\n{im_file}\n')
        #
        #     # Copy files
        #     copy2(im_file, im_file_out)
        #     lab = nib.load(lab_file)
        #     nib.save(lab, lab_file_out)
        #     msk = nib.load(msk_file)
        #     nib.save(msk, msk_file_out)
        #
        # # Copy simulations
        # for im_file in im_files_sims:
        #     nn += 1
        #     # Get original files
        #     bpath = os.path.split(im_file)[0]
        #     lab_file = os.path.join(bpath, lab_tmp)
        #     msk_file = os.path.join(bpath, msk_tmp)
        #
        #     # Create output files
        #     spath_inner = os.path.join(spath_train, f'{nn:02d}')
        #     if not os.path.exists(spath_inner):
        #         os.mkdir(spath_inner)
        #     im_file_out = os.path.join(spath_inner, im_tmp)
        #     lab_file_out = os.path.join(spath_inner, lab_tmp_save)
        #     msk_file_out = os.path.join(spath_inner, msk_tmp_save)
        #     txt_file_out = os.path.join(spath_inner, txt_tmp)
        #     with open(txt_file_out, 'w') as f:
        #         f.write(f'Simulated data\n{im_file}\n')
        #
        #     # Copy files
        #     copy2(im_file, im_file_out)
        #     lab = nib.load(lab_file)
        #     nib.save(lab, lab_file_out)
        #     msk = nib.load(msk_file)
        #     nib.save(msk, msk_file_out)
        #
        # # Move testing data
        # for nn, ind in enumerate(test_ind):
        #     # Get original files
        #     im_file = im_files[ind]
        #     bpath = os.path.split(im_file)[0]
        #     lab_file = os.path.join(bpath, lab_tmp)
        #     msk_file = os.path.join(bpath, msk_tmp)
        #
        #     # Create output files
        #     spath_inner = os.path.join(spath_test, f'{nn:02d}')
        #     if not os.path.exists(spath_inner):
        #         os.mkdir(spath_inner)
        #     im_file_out = os.path.join(spath_inner, im_tmp)
        #     lab_file_out = os.path.join(spath_inner, lab_tmp_save)
        #     msk_file_out = os.path.join(spath_inner, msk_tmp_save)
        #     txt_file_out = os.path.join(spath_inner, txt_tmp)
        #     with open(txt_file_out, 'w') as f:
        #         f.write(f'Real data\n{im_file}\n')
        #
        #     # Copy files
        #     copy2(im_file, im_file_out)
        #     lab = nib.load(lab_file)
        #     nib.save(lab, lab_file_out)
        #     msk = nib.load(msk_file)
        #     nib.save(msk, msk_file_out)


    print('Done')


def prepare_train_data():

    output_path = 'data/kfold'
    xfer_start = './tmp/ckpt_xfer_210305/checkpoint-8212*'
    checkpoint_file = 'tmp/ckpt_201102/checkpoint-latest'
    xfer_files = glob(xfer_start)

    # Configuration file
    base_confile = 'conf_kfold_sim_real_00.json'
    confile = 'conf_kfold_train_sim_real_eval_{:02d}.json'

    # Load base
    with open(base_confile, 'r') as f:
        base_df = json.load(f)

    # Get all data files
    base_im_path = 'data/kfold/train_sim_real'
    im_files = glob(os.path.join(base_im_path, '**', 'image.nii'))

    # set up Kfold object
    kf = KFold(n_splits=5, shuffle=False)

    for n, _ in enumerate(kf.split(im_files)):
        print(f'Working on split {n:d}')
        # Set up output path
        spath_train = os.path.join(output_path, 'train_for_sim', f'{n+1:02d}')
        spath_test = os.path.join(output_path, 'test_sim_real', f'{n+1:02d}')
        if not os.path.exists(spath_train):
            os.mkdir(spath_train)
        if not os.path.exists(spath_test):
            os.mkdir(spath_test)

        # Create testing configuration file
        df = base_df.copy()
        df['TrainingSetting']['Restore'] = False
        df['TrainingSetting']['Data']['TestingDataDirectory'] = os.path.join('./', spath_test)
        df['TrainingSetting']['LogDir'] = os.path.join('./', f'tmp/log_kfold/{n+1:02d}')
        df['TrainingSetting']['CheckpointDir'] = os.path.join('./', f'tmp/ckpt_kfold/{n+1:02d}')
        df['EvaluationSetting']['Data']['EvaluateDataDirectory'] = os.path.join('./', spath_train)

        file = confile.format(n)
        with open(file, 'w') as f:
            json_str = json.dumps(df, indent=4)
            f.write(json_str)

    print('Done')


def run_training_data():
    prepare_train_data()
    current_dir = os.getcwd()

    confile = 'conf_kfold_train_sim_real_eval_??.json'

    # Get config files
    config_files = sorted(glob(confile))
    log_dir = 'logs'
    gpu = 1
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for config in config_files:
        print('Running evaluations for {:s}'.format(config))
        # Update checkpoint
        with open(config, 'r') as f:
            df = json.load(f)

        ckp_path = df['TrainingSetting']['CheckpointDir']
        ckps = sorted(glob(os.path.join(ckp_path, 'checkpoint*')))
        ckp = os.path.join(ckp_path, os.path.split(os.path.splitext(ckps[-2])[0])[1])
        df['EvaluationSetting']['CheckpointPath'] = ckp

        # Save configuration
        with open(config, 'w') as f:
            json_str = json.dumps(df, indent=4)
            f.write(json_str)

        output_file = os.path.join(log_dir, os.path.splitext(config)[0] + '_output_eval_train_data.txt')
        cmd = 'docker run --rm -it --gpus "device={:d}" ' \
              '-v {:s}:/home/app ' \
              'lungdl python main.py -p evaluate --config_json {:s} > {:s}'.format(gpu, current_dir, config, output_file)
        call(cmd, shell=True)
        print('Finished {:s}\n'.format(config))


def train(configs, gpu):
    current_dir = os.getcwd()

    for config in configs:
        print('Starting {:s}\n'.format(config))
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        output_file = os.path.join(log_dir, os.path.splitext(config)[0] + 'output.txt')
        cmd = 'docker run --rm -it --gpus "device={:d}" ' \
              '-v {:s}:/home/app ' \
              '-it lungdl python main.py --config_json {:s} > {:s}'.format(gpu, current_dir, config, output_file)
        call(cmd, shell=True)

        print('Running evaluations for {:s}'.format(config))
        # Update checkpoint
        with open(config, 'r') as f:
            df = json.load(f)

        ckp_path = df['TrainingSetting']['CheckpointDir']
        ckps = sorted(glob(os.path.join(ckp_path, 'checkpoint*')))
        ckp = os.path.join(ckp_path, os.path.split(os.path.splitext(ckps[-2])[0])[1])
        df['EvaluationSetting']['CheckpointPath'] = ckp

        # Save configuration
        with open(config, 'w') as f:
            json_str = json.dumps(df, indent=4)
            f.write(json_str)

        output_file = os.path.join(log_dir, os.path.splitext(config)[0] + 'output_eval.txt')
        cmd = 'docker run --rm -it --gpus "device={:d}" ' \
              '-v {:s}:/home/app ' \
              '-it lungdl python main.py -p evaluate --config_json {:s} > {:s}'.format(gpu, current_dir, config, output_file)
        call(cmd, shell=True)
        print('Finished {:s}\n'.format(config))


def run_kfold():

    # Set up parameters
    num_gpus = 1
    confile = 'conf_kfold_sim_real_??.json'

    # Get config files
    config_files = sorted(glob(confile))
    # train(config_files, 0)

    # Create task list
    tasks = [list() for _ in range(num_gpus)]
    for z in range(len(config_files)):
        thr = z % num_gpus
        tasks[thr].append(config_files[z])

    thread = [list() for _ in range(num_gpus)]
    for z in range(num_gpus):
        thread[z] = Thread(target=train, args=[tasks[z], z+1])
        thread[z].start()

    for z in range(num_gpus):
        thread[z].join()

    print('\n\nAll finished\n\n')


if __name__ == "__main__":

    # prepare_data()
    # run_kfold()
    run_training_data()
