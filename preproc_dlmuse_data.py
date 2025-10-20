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


def get_fsl_dir():
    fsl_dir = conf.config['fsl']['path']
    if not os.path.isdir(fsl_dir):
        raise FileNotFoundError
    return fsl_dir

def get_fsl_script_prefix():
    return '#!/bin/bash \nexport FSLVERSION=6.0.5.2 || exit -1 \n'


class BaseT1Preproc:
    '''Base class. 
       Gets data paths for model input
       Using ADNI DLMUSE segmentation data'''

    def __init__(self):
        self.datetime = datetime.now().strftime("%Y.%m.%d.%H%M.%S")
        self.base_dir = os.getcwd()
        self.iopaths = os.path.join(self.base_dir, 'iopaths')
        self.data_dir = os.path.join(self.base_dir, 'data', 'DLMUSE_ADNI')
        self.brain_mask_dir = os.path.join(self.data_dir, 'brain_mask')
        self.brain_dir = os.path.join(self.data_dir, 'brain')
        self.wm_mask_dir = os.path.join(self.data_dir, 'wm_mask')
        self.wm_dir = os.path.join(self.data_dir, 'wm')
        self.reg_dir = os.path.join(self.data_dir, 'regtoMNI')
        self.zscore_dir = os.path.join(self.data_dir, 'zscore_wm_norm')
        self.T1_paths = [line.replace('\n','') for line in open(os.path.join(self.iopaths, 'adni4_dlmuse_input_paths.txt'), 'r')]
        self.seg_paths = [line.replace('\n','') for line in open(os.path.join(self.iopaths, 'adni4_dlmuse_seg_paths.txt'), 'r')]
        self.atlas = os.path.join(self.base_dir, 'atlases', 'MNI152_T1_1mm_brain.nii.gz')
    
    def load_nii(self, fname):
        try:
            img = nib.load(fname)
            #img = nib.as_closest_canonical(img)
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
    """
    Preprocessing ADNI T1 DLMUSE data.
    Use segmentation to extract brain and WM volume.
    Register brain to template.
    Normalize brain by z-score using WM volume as reference.
    """

    def __init__(self):
        super(PreprocDataset, self).__init__()

    def create_brain_mask(self, code, seg):
        """Create mask of brain volume (nonzero voxels) from seg"""
        dat, aff = self.load_nii(seg)
        mask = dat != 0
        mask = mask.astype(np.uint8)
        brain_mask = os.path.join(self.brain_mask_dir, code + '_brain_mask.nii.gz')
        self.save_nii(mask, aff, brain_mask)
    
    def extract_brain(self, code, img):
        """Extract brain from T1 using brain mask"""
        mask = os.path.join(self.brain_mask_dir, code + '_brain_mask.nii.gz')
        brain = os.path.join(self.brain_dir, code + '_brain.nii.gz')
        cmd = f'fslmaths {img} -mul {mask} {brain}'
        self.run_cmd(cmd, 'extract-brain')
    
    def create_wm_mask(self, code, seg):
        """Create mask of WM volume (voxel intensities 81-88) from seg"""
        dat, aff = self.load_nii(seg)
        wm_vals = [81,82,83,84,85,86,87,88]
        mask = np.isin(dat, wm_vals)
        mask = mask.astype(np.uint8)
        wm_mask = os.path.join(self.wm_mask_dir, code + '_wm_mask.nii.gz')
        self.save_nii(mask, aff, wm_mask)
    
    def extract_wm(self, code, img):
        """Extract WM from T1 using WM mask"""
        mask = os.path.join(self.wm_mask_dir, code + '_wm_mask.nii.gz')
        wm = os.path.join(self.wm_dir, code + '_wm.nii.gz')
        cmd = f'fslmaths {img} -mul {mask} {wm}'
        self.run_cmd(cmd, 'extract-wm')
    
    def reg_to_template(self, code):
        """Linear register T1 brain to MNI atlas with 12 DOF"""
        brain = os.path.join(self.brain_dir, code + '_brain.nii.gz')
        mni = os.path.join(self.reg_dir, code + '_brain_MNI.nii.gz')
        cmd = f'{get_fsl_script_prefix()}' \
              f'flirt -in {brain} -ref {self.atlas} -dof 12 -out {mni}'  
        self.run_cmd(cmd, 'reg-t1brain-mni')
    
    def zscore_normalize(self, code):
        """Z-score normalize registered brain, using WM as reference"""
        brain_mni = os.path.join(self.reg_dir, code + '_brain_MNI.nii.gz')
        wm = os.path.join(self.wm_dir, code + '_wm.nii.gz')
        img_norm = os.path.join(self.zscore_dir, code + '.nii.gz')
        brain_dat, brain_aff = self.load_nii(brain_mni)
        wm_dat, wm_aff = self.load_nii(wm)
        mask = brain_dat != 0
        wm_mask = wm_dat != 0
        mean = np.mean(wm_mask)
        std = np.std(wm_mask)
        if std == 0:
            std = 1
        norm = np.zeros_like(brain_dat)
        norm[mask] = (brain_dat[mask]-mean)/std
        self.save_nii(norm, brain_aff, img_norm)
    
    def run(self):
        logger.info('Preprocessing pipeline initiated')
        for img, seg in zip(self.T1_paths, self.seg_paths):
            code = os.path.basename(img).replace('.T1.nii.gz','')
            #code = img.split('/')[-3]
            if code + '.nii.gz' not in os.listdir(self.zscore_dir):
                logger.info('Preprocessing image ' + code)
                try:
                    self.create_brain_mask(code, seg)
                    self.extract_brain(code, img)
                    self.create_wm_mask(code, seg)
                    self.extract_wm(code, img)
                    self.reg_to_template(code)
                    self.zscore_normalize(code)
                except FileNotFoundError:
                    logger.info('DLMUSE input file(s) missing for ' + code + '. Skipping this case.')
                    continue
                except FailedScriptError:
                    logger.info('Script failed for ' + code + '. Skipping this case.')
                    continue


if __name__=='__main__':
    p = PreprocDataset()
    p.run()

