import os, shutil
import string, csv, json, subprocess, logging
from datetime import datetime
import numpy as np
import pandas as pd
import nibabel as nib

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
    return conf.config['afni']['24']

def get_fsl_script_prefix():
    return '#!/bin/bash \nexport FSLVERSION=6.0.5.2 || exit -1 \n'

def get_freesurfer_script_prefix():
    return '#!/bin/bash \nexport FSVERSION=7.3.2 || exit -1 \n'


class BaseT1Preproc:
    '''Base class. 
       Gets data paths for model input'''

    def __init__(self):
        self.datetime = datetime.now().strftime("%Y.%m.%d.%H%M.%S")
        self.base_dir = os.getcwd()
        self.iopaths = os.path.join(self.base_dir, 'iopaths')
        self.temp_dir = os.path.join(self.base_dir, 'temp_data')
        self.data_dir = os.path.join(self.base_dir, 'data', 'regtoMNI')
        self.zscore_dir = os.path.join(self.base_dir, 'data', 'zscore_norm')
        self.T1_paths = [line.replace('\n','') for line in open(os.path.join(self.iopaths, 'adni_t1s_reg_split.txt'), 'r')]
        self.atlas = os.path.join(self.base_dir, 'atlases', 'MNI152_T1_1mm_brain.nii.gz')
    
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


class PreprocDataset(BaseT1Preproc):
    '''Converts to nifti and stores in data folder'''

    def __init__(self):
        super(PreprocDataset, self).__init__()

    def mgz_to_nii(self, code, img):
        '''Convert mgz to nifti file'''
        orig_path = img.replace('brain.mgz','orig/001.mgz')
        nii_temp = os.path.join(self.temp_dir, code + '.nii.gz')
        cmd = f'{get_freesurfer_script_prefix()}' \
              f'mri_convert -rl {orig_path} -rt nearest {img} {nii_temp}'
        self.run_cmd(cmd, 'conv-to-nii')
        resamp = os.path.join(get_afni(), '3dresample')
        cmd = f'{resamp} -orient LPI -input {nii_temp} -prefix {nii_temp} -overwrite'
        self.run_cmd(cmd, 'resample-t1')
    
    def reg_to_template(self, code, img=None):
        '''Linear register T1 to MNI atlas with 12 DOF'''
        if img is not None:
            img = os.path.join(self.temp_dir, code + '.nii.gz')
        reg_img = os.path.join(self.data_dir, code + '.nii.gz')
        cmd = f'{get_fsl_script_prefix()}' \
              f'flirt -in {img} -ref {self.atlas} -dof 12 -out {reg_img}'  
        self.run_cmd(cmd, 'reg-t1')
    
    def zscore_normalize(self, code, img=None):
        '''Z-score normalization on brain voxels only'''
        if img is None:
            img = os.path.join(self.data_dir, code + '.nii.gz')
        img_norm = os.path.join(self.zscore_dir, code + '.nii.gz')
        dat, aff = self.load_nii(img)
        mask = dat != 0
        brain = dat[mask]
        mean = np.mean(brain)
        std = np.std(brain)
        if std == 0:
            std = 1
        norm = np.zeros_like(dat)
        norm[mask] = (dat[mask]-mean)/std
        self.save_nii(norm, aff, img_norm)
    
    def run(self):
        logger.info('Preprocessing pipeline initiated')
        for img in self.T1_paths:
            #code = img.split('/')[-3]
            code = os.path.basename(img).replace('.nii.gz','')
            if code + '.nii.gz' not in os.listdir(self.zscore_dir):
                logger.info('Preprocessing image ' + code)
                try:
                    #self.mgz_to_nii(code, img)
                    #self.reg_to_template(code)
                    self.zscore_normalize(code, img)
                except FailedScriptError:
                    logger.info('Script failed for image ' + code)
                    continue


if __name__=='__main__':
    t = BaseT1Preproc()
    p = PreprocDataset()
    p.run()

