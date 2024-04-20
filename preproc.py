import os, shutil
import string, csv, json, subprocess, logging
from datetime import datetime
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# LOCAL IMPORTS
from utils import conf
from utils.exceptions import FailedScriptError

logger = logging.getLogger(__name__)


def get_freesurfer():
    fs_path = conf.config["freesurfer"]["7.3.2"]
    os.environ['FSVERSION'] = '7.3.2'
    return fs_path

def get_fsl_dir():
    fsl_dir = conf.config['fsl']['path']
    if not os.path.isdir(fsl_dir):
        raise FileNotFoundError
    return fsl_dir

def get_afni():
    return conf.config['afni']['AFNIPATH']

def get_ants_apply():
    antspath = conf.config['ants']['ANTSPATH']
    os.environ['ANTSPATH'] = antspath
    os.environ['ANTSVERSION'] = conf.config['ants']['ANTSVERSION']
    return os.path.join(antspath, 'antsApplyTransforms')

def get_fsl_script_prefix():
    return '#!/bin/bash \nexport FSLVERSION=6.0.5.2 || exit -1 \n'

def get_freesurfer_script_prefix():
    return '#!/bin/bash \nexport FSVERSION=7.3.2 || exit -1 \n'


class BaseT1CAE:
    '''Base class. 
       Gets data paths for model input'''

    def __init__(self):
        self.datetime = datetime.now().strftime("%Y.%m.%d.%H%M.%S")
        self.base_dir = os.getcwd()
        # paths for image data
        self.T1_list = [line.replace('\n','') for line in open('ADNI_T1s.txt', 'r')]
        self.data_temp_dir = os.path.join(self.base_dir, 'temp_data')
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.paths = self.get_T1_brain_paths_from_FS()
        self.niftis = []
        # split train/test data
    
    def get_T1_brain_paths_from_FS(self):
        path_dict = {}
        for t in self.T1_list:
            if t.startswith('0') or t.startswith('1') or t.startswith('9'):
                proj_name = 'ADNI_v43Long'
            elif t.startswith('ADNI1'):
                proj_name = 'ADNI_3T_FSv51'
            elif t.startswith('ADNI2'):
                proj_name = 'ADNI2_v51'
            elif t.startswith('ADNI3'):
                proj_name = 'ADNI3_FSdn'
            mgz = '/m/InProcess/External/' + proj_name + '/Freesurfer/subjects/' + t + '/mri/brain.mgz'
            path_dict[t] = mgz
        return path_dict
    
    def split_train_test(self):
        # 80/20 train/test split
        pass
    
    def load_nii(self, fname):
        try:
            img = nib.load(fname)
            img = nib.as_closest_canonical(img)
            dat = img.get_fdata()
            aff = img.affine
            return dat, aff
        except FileNotFoundError as e:
            logger.info(str(e))
            self.outcome = 'Error'
            self.comments = str(e)
            raise e

    def save_nii(self, dat, aff, fname):
        img = nib.Nifti1Image(dat, aff)
        img.to_filename(fname)

    def run_cmd(self, cmd, script_name=None):
        if script_name is None:
            script_name = f'{self.datetime}-{self.__class__.__name__}.sh'
        else:
            script_name = f'{self.datetime}-{script_name}.sh'
        logger.debug(cmd)    
        script_path = self.prep_cmd_script(cmd, script_name)
        output = self.run_script(script_path)
        return output

    def run_script(self, script_path):
        p = subprocess.Popen(script_path,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             encoding="utf-8",errors='ignore',shell=True,env=os.environ)
        logger.info(f"running script:{script_path}")
        output, error = p.communicate()
        if output:
            logger.info(output)
        if error and p.returncode == 0:
            logger.warning(error)
        if error and p.returncode != 0:
            logger.error(output)
            logger.error(error)
        if p.returncode != 0:
            lines = []
            if output:
                lines = output.splitlines()
            if error:
                lines = lines + error.splitlines()
            if len(lines) > 20:
                lines = lines[-20]
            logger.error(f"Script Failed! {script_path}")
            raise FailedScriptError(f"{script_path} failed")
        return output

    def prep_cmd_script(self, cmd, script_name):
        script_template = string.Template("$cmd")
        script = script_template.safe_substitute({'cmd': cmd})
        script_path = os.path.join(self.base_dir, 'scripts', script_name)
        if not os.path.exists(os.path.dirname(script_path)):
            os.makedirs(os.path.dirname(script_path))
        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o777)
        return script_path


class PreprocDataset(BaseT1CAE):
    '''Converts to nifti and stores in data folder'''

    def __init__(self):
        super(PreprocDataset, self).__init__()

    def mgz_to_nii(self, img):
        mgz_path = self.paths[img]
        orig_path = mgz_path.replace('brain.mgz','orig/001.mgz')
        nii_temp = os.path.join(self.data_temp_dir, img + '.nii.gz')
        nii_img = os.path.join(self.data_dir, img + '.nii.gz')
        cmd = f'{get_freesurfer_script_prefix()}' \
              f'mri_convert -rl {orig_path} -rt nearest {mgz_path} {nii_temp}'
        self.run_cmd(cmd, 'conv-to-nii')
        resamp = os.path.join(get_afni(), '3dresample')
        cmd = f'{resamp} -orient LPI -input {nii_temp} -prefix {nii_img} -overwrite'
        self.run_cmd(cmd, 'resample-t1')
    
    def run(self):
        logger.info('Converting mgz images to nii format')
        for img in self.paths.keys():
            logger.info('Converting image ' + img)
            try:
                self.mgz_to_nii(img)
            except FailedScriptError:
                continue


if __name__=='__main__':
    t = BaseT1CAE()
    p = PreprocDataset()
    p.run()

