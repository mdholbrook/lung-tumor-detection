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
    xfer_files = glob(xfer_start)

    # Configuration file
    base_confile = 'config_MH_xfer.json'
    confile_xfer = 'conf_kfold_xfer_{:02d}.json'
    confile = 'conf_kfold_{:02d}.json'

    # Load base
    with open(base_confile, 'r') as f:
        base_df = json.load(f)
    
    # Get all data files
    base_im_path = 'data/Real_xfer_210305'
    im_files = glob(os.path.join(base_im_path, '**', 'image.nii'))
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
        spath_train = os.path.join(output_path, 'train', f'{n+1:02d}')
        spath_test = os.path.join(output_path, 'test', f'{n+1:02d}')
        spath_test_xfer = os.path.join(output_path, 'test_xfer', f'{n+1:02d}')
        if not os.path.exists(spath_train):
            os.mkdir(spath_train)
        if not os.path.exists(spath_test):
            os.mkdir(spath_test)

        # Create training configuration file
        df_xfer = base_df.copy()
        df_xfer['TrainingSetting']['Data']['TrainingDataDirectory'] = os.path.join('./', spath_train)
        df_xfer['TrainingSetting']['Data']['TestingDataDirectory'] = os.path.join('./', spath_test_xfer)
        df_xfer['TrainingSetting']['LogDir'] = os.path.join('./', f'tmp/log_kfold_xfer/{n+1:02d}')
        df_xfer['EvaluationSetting']['Data']['EvaluateDataDirectory'] = os.path.join('./', spath_test_xfer)
        ckpt = os.path.join('./', f'tmp/ckpt_kfold_xfer/{n+1:02d}')
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        for xfer_file in xfer_files:
            copy2(xfer_file, ckpt)
        copy2(checkpoint_file, ckpt)
        df_xfer['TrainingSetting']['CheckpointDir'] = ckpt
        
        file = confile_xfer.format(n)
        with open(file, 'w') as f:
            json_str = json.dumps(df_xfer, indent=4)
            f.write(json_str)

        # Create testing configuration file
        df = df_xfer.copy()
        df['TrainingSetting']['Restore'] = False
        df['TrainingSetting']['Data']['TestingDataDirectory'] = os.path.join('./', spath_test)
        df['TrainingSetting']['LogDir'] = os.path.join('./', f'tmp/log_kfold/{n+1:02d}')
        df['TrainingSetting']['CheckpointDir'] = os.path.join('./', f'tmp/ckpt_kfold/{n+1:02d}')
        df['EvaluationSetting']['Data']['EvaluateDataDirectory'] = os.path.join('./', spath_test)

        file = confile.format(n)
        with open(file, 'w') as f:
            json_str = json.dumps(df, indent=4)
            f.write(json_str)
                    
        # Move training data
        for nn, ind in enumerate(train_ind):
            # Get original files
            im_file = im_files[ind]
            bpath = os.path.split(im_file)[0]
            lab_file = os.path.join(bpath, lab_tmp)
            msk_file = os.path.join(bpath, msk_tmp)
            txt_file = os.path.join(bpath, txt_tmp)

            # Create output files
            spath_inner = os.path.join(spath_train, f'{nn:02d}')
            if not os.path.exists(spath_inner):
                os.mkdir(spath_inner)
            im_file_out = os.path.join(spath_inner, im_tmp)
            lab_file_out = os.path.join(spath_inner, lab_tmp_save)
            msk_file_out = os.path.join(spath_inner, msk_tmp_save)
            txt_file_out = os.path.join(spath_inner, txt_tmp)

            # Copy files
            copy2(im_file, im_file_out)
            copy2(txt_file, txt_file_out)
            lab = nib.load(lab_file)
            nib.save(lab, lab_file_out)
            msk = nib.load(msk_file)
            nib.save(msk, msk_file_out)

        # Move testing data
        for nn, ind in enumerate(test_ind):
            # Get original files
            im_file = im_files[ind]
            bpath = os.path.split(im_file)[0]
            lab_file = os.path.join(bpath, lab_tmp)
            msk_file = os.path.join(bpath, msk_tmp)
            txt_file = os.path.join(bpath, txt_tmp)

            # Create output files
            spath_inner = os.path.join(spath_test, f'{nn:02d}')
            if not os.path.exists(spath_inner):
                os.mkdir(spath_inner)
            im_file_out = os.path.join(spath_inner, im_tmp)
            lab_file_out = os.path.join(spath_inner, lab_tmp_save)
            msk_file_out = os.path.join(spath_inner, msk_tmp_save)
            txt_file_out = os.path.join(spath_inner, txt_tmp)

            # Copy files
            copy2(im_file, im_file_out)
            copy2(txt_file, txt_file_out)
            lab = nib.load(lab_file)
            nib.save(lab, lab_file_out)
            msk = nib.load(msk_file)
            nib.save(msk, msk_file_out)

    # Copy for xfer
    print('Copying test data for transfer learning')
    spath_test = os.path.join(output_path, 'test')
    spath_test_xfer = os.path.join(output_path, 'test_xfer')
    if os.path.exists(spath_test_xfer):
        os.rmdir(spath_test_xfer)
    copytree(spath_test, spath_test_xfer)
    print('Done')


def prepare_train_data():

    output_path = 'data/kfold'
    xfer_start = './tmp/ckpt_xfer_210305/checkpoint-8212*'
    checkpoint_file = 'tmp/ckpt_201102/checkpoint-latest'
    xfer_files = glob(xfer_start)

    # Configuration file
    base_confile = 'config_MH_xfer.json'
    confile_xfer = 'conf_kfold_xfer_train_eval_{:02d}.json'
    confile = 'conf_kfold_train_eval_{:02d}.json'

    # Load base
    with open(base_confile, 'r') as f:
        base_df = json.load(f)

    # Get all data files
    base_im_path = 'data/Real_xfer_210305'
    im_files = glob(os.path.join(base_im_path, '**', 'image.nii'))
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
        spath_train = os.path.join(output_path, 'train', f'{n+1:02d}')
        spath_train_xfer = os.path.join(output_path, 'train_xfer', f'{n+1:02d}')
        spath_test = os.path.join(output_path, 'test', f'{n+1:02d}')
        spath_test_xfer = os.path.join(output_path, 'test_xfer', f'{n+1:02d}')
        if not os.path.exists(spath_train):
            os.mkdir(spath_train)
        # if not os.path.exists(spath_train_xfer):
        #     os.makedirs(spath_train_xfer)
        if not os.path.exists(spath_test):
            os.mkdir(spath_test)

        # Create training configuration file
        df_xfer = base_df.copy()
        df_xfer['TrainingSetting']['Data']['TrainingDataDirectory'] = os.path.join('./', spath_train)
        df_xfer['TrainingSetting']['Data']['TestingDataDirectory'] = os.path.join('./', spath_test_xfer)
        df_xfer['TrainingSetting']['LogDir'] = os.path.join('./', f'tmp/log_kfold_xfer/{n+1:02d}')
        df_xfer['EvaluationSetting']['Data']['EvaluateDataDirectory'] = os.path.join('./', spath_train_xfer)
        ckpt = os.path.join('./', f'tmp/ckpt_kfold_xfer/{n+1:02d}')
        if not os.path.exists(ckpt):
            os.makedirs(ckpt)
        df_xfer['TrainingSetting']['CheckpointDir'] = ckpt

        file = confile_xfer.format(n)
        with open(file, 'w') as f:
            json_str = json.dumps(df_xfer, indent=4)
            f.write(json_str)

        # Create testing configuration file
        df = df_xfer.copy()
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

    confile_xfer = 'conf_kfold_xfer_train_eval_??.json'
    confile = 'conf_kfold_train_eval_??.json'

    # Get config files
    config_files = sorted(glob(confile_xfer)) + sorted(glob(confile))
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
              '-v {:s}:/home/app '\
              '-it lungdl python main.py --config_json {:s} > {:s}'.format(gpu, current_dir, config, output_file)
        # call(cmd, shell=True)

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
    num_gpus = 2
    confile_xfer = 'conf_kfold_xfer_??.json'
    confile = 'conf_kfold_??.json'

    # Get config files
    config_files = sorted(glob(confile_xfer)) + sorted(glob(confile))
    # train(config_files, 0)

    # Create task list
    tasks = [list() for _ in range(num_gpus)]
    for z in range(len(config_files)):
        thr = z % num_gpus
        tasks[thr].append(config_files[z])

    thread = [list() for _ in range(num_gpus)]
    for z in range(num_gpus):
        thread[z] = Thread(target=train, args=[tasks[z], z])
        thread[z].start()

    for z in range(num_gpus):
        thread[z].join()

    print('\n\nAll finished\n\n')
    
    
if __name__ == "__main__":

    run_training_data()
    # prepare_data()
    run_kfold()
