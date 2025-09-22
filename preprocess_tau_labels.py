import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import statistics as stats
import datetime as dt 

def within_timeperiod(date1, date1_format, date2, date2_format, ndays):
	"""Checks that date1 and date2 are no more than ndays apart"""
	date1=dt.datetime.strptime(date1, date1_format)
	date2=dt.datetime.strptime(date2, date2_format)
	backward = date1 - dt.timedelta(days=ndays)
	forward = date1 + dt.timedelta(days=ndays)
	if date2 >= backward and date2 <= forward:
		return True
	else:
		return False		

# raw data files
tau_pvc = pd.read_csv('encodings/covar_data/TauPVC_6mm_PET.csv')
adni_data = pd.read_csv('encodings/data.csv')

# ROI volumes from tau_pvc
medtemp_vols = ['CTX_LH_ENTORHINAL_VOLUME','CTX_RH_ENTORHINAL_VOLUME',
				'CTX_LH_PARAHIPPOCAMPAL_VOLUME','CTX_RH_PARAHIPPOCAMPAL_VOLUME',
				'LEFT_AMYGDALA_VOLUME','RIGHT_AMYGDALA_VOLUME']
mettemp_vols = ['CTX_LH_ENTORHINAL_VOLUME','CTX_RH_ENTORHINAL_VOLUME',
				'CTX_LH_PARAHIPPOCAMPAL_VOLUME','CTX_RH_PARAHIPPOCAMPAL_VOLUME',
				'LEFT_AMYGDALA_VOLUME','RIGHT_AMYGDALA_VOLUME',
				'CTX_LH_FUSIFORM_VOLUME','CTX_RH_FUSIFORM_VOLUME',
				'CTX_LH_INFERIORTEMPORAL_VOLUME','CTX_RH_INFERIORTEMPORAL_VOLUME',
				'CTX_LH_MIDDLETEMPORAL_VOLUME','CTX_RH_MIDDLETEMPORAL_VOLUME']
temppar_vols = ['CTX_LH_BANKSSTS_VOLUME','CTX_RH_BANKSSTS_VOLUME',
				'CTX_LH_CUNEUS_VOLUME','CTX_RH_CUNEUS_VOLUME',
				'CTX_LH_INFERIORPARIETAL_VOLUME','CTX_RH_INFERIORPARIETAL_VOLUME',
				'CTX_LH_SUPERIORPARIETAL_VOLUME','CTX_RH_SUPERIORPARIETAL_VOLUME',
				'CTX_LH_INFERIORTEMPORAL_VOLUME','CTX_RH_INFERIORTEMPORAL_VOLUME',
				'CTX_LH_MIDDLETEMPORAL_VOLUME','CTX_RH_MIDDLETEMPORAL_VOLUME',
				'CTX_LH_SUPERIORTEMPORAL_VOLUME','CTX_RH_SUPERIORTEMPORAL_VOLUME',
				'CTX_LH_ISTHMUSCINGULATE_VOLUME','CTX_RH_ISTHMUSCINGULATE_VOLUME',
				'CTX_LH_LATERALOCCIPITAL_VOLUME','CTX_RH_LATERALOCCIPITAL_VOLUME',
				'CTX_LH_LINGUAL_VOLUME','CTX_RH_LINGUAL_VOLUME',
				'CTX_LH_POSTERIORCINGULATE_VOLUME','CTX_RH_POSTERIORCINGULATE_VOLUME',
				'CTX_LH_PRECUNEUS_VOLUME','CTX_RH_PRECUNEUS_VOLUME',
				'CTX_LH_SUPRAMARGINAL_VOLUME','CTX_RH_SUPRAMARGINAL_VOLUME']
all_vols = list(set(medtemp_vols) | set(mettemp_vols) | set(temppar_vols))

# composite ROI components
medtemp_rois = [i.replace('VOLUME','SUVR') for i in medtemp_vols]
mettemp_rois = [i.replace('VOLUME','SUVR') for i in mettemp_vols]
temppar_rois = [i.replace('VOLUME','SUVR') for i in temppar_vols]
all_rois = list(set(medtemp_rois) | set(mettemp_rois) | set(temppar_rois))

# mapping of ROI to volume
roi_vols = {all_rois[i]: all_rois[i].replace('SUVR','VOLUME') for i in range(len(all_rois))}
cols = all_rois + all_vols
cols.insert(0, 'PTID')
cols.insert(1, 'SCANDATE')
df = pd.DataFrame(tau_pvc, columns=cols)

# compute total weighted SUVR/volume values and add to dataframe
all_total_vols = []
all_total_suvrs = []
for i in df.index:
	total_vols = []
	total_suvrs = []
	for mroi in [medtemp_rois, mettemp_rois, temppar_rois]:
		vol = 0
		suvr = 0
		for roi in mroi:
			v = roi_vols[roi]
			vol += df.at[i, v]
			suvr += df.at[i, roi] * df.at[i, v]
		total_vols.append(vol)
		total_suvrs.append(suvr)
	all_total_vols.append(total_vols)
	all_total_suvrs.append(total_suvrs)	

medtemp_suvrs = [i[0] for i in all_total_suvrs]
medtemp_totalvols = [i[0] for i in all_total_vols]
mettemp_suvrs = [i[1] for i in all_total_suvrs]
mettemp_totalvols = [i[1] for i in all_total_vols]
temppar_suvrs = [i[2] for i in all_total_suvrs]
temppar_totalvols = [i[2] for i in all_total_vols]

df.insert(1, 'MedTempWeightedSUVR', medtemp_suvrs)
df.insert(2, 'MedTempTotalVol', medtemp_totalvols)
df.insert(3, 'MetTempWeightedSUVR', mettemp_suvrs)
df.insert(4, 'MetTempTotalVol', mettemp_totalvols)
df.insert(5, 'TempParWeightedSUVR', temppar_suvrs)
df.insert(6, 'TempParTotalVol', temppar_totalvols)

# create final dataframe of weighted avg tau SUVR
data = pd.DataFrame(df, columns=['PTID','MedTempWeightedSUVR','MedTempTotalVol','MetTempWeightedSUVR','MetTempTotalVol','TempParWeightedSUVR','TempParTotalVol','SCANDATE'])
medtemp_tau = []
mettemp_tau = []
temppar_tau = []
for i in data.index:
	medtemp_tau.append(data.at[i,'MedTempWeightedSUVR'] / data.at[i,'MedTempTotalVol'])
	mettemp_tau.append(data.at[i,'MetTempWeightedSUVR'] / data.at[i,'MetTempTotalVol'])
	temppar_tau.append(data.at[i,'TempParWeightedSUVR'] / data.at[i,'TempParTotalVol'])

data.insert(1, 'MedTempTau', medtemp_tau)
data.insert(2, 'MetTempTau', mettemp_tau)
data.insert(3, 'TempParTau', temppar_tau)

tau_data = pd.DataFrame(data, columns=['PTID','MedTempTau','MetTempTau','TempParTau','SCANDATE'])
tau_data.to_csv('encodings/covar_data/tau_metaROI_SUVR.csv', index=False)

# get tau SUVR with available ADNI T1 data
df0 = pd.merge(adni_data, tau_data, how='inner', on=['PTID'])
ndays = 365
fmt = '%Y-%m-%d'
for i in df0.index:
    mridate = df0.at[i,'ScanDate']
    taudate = df0.at[i,'SCANDATE']
    if within_timeperiod(mridate,fmt,taudate,fmt,ndays):
        pass
    else:
        df0 = df0.drop(i)
df0 = df0.reset_index()
df0 = df0.drop(columns=['index','SCANDATE'])
df0.to_csv('encodings/data_tau.csv')

# plot data
def plot_data(nbins, filepath, show=False):
	fig, ax = plt.subplots(1, 3, sharey=True)
	ax[0].hist(tau_data['MedTempTau'], bins=nbins)
	ax[0].set_title('Medial temporal')
	ax[1].hist(tau_data['MetTempTau'], bins=nbins)
	ax[1].set_title('Metatemporal')
	ax[2].hist(tau_data['TempParTau'], bins=nbins)
	ax[2].set_title('Temporoparietal')
	fig.suptitle('Data distributions')
	if show:
		plt.show()
	plt.savefig(filepath)

plot_data(nbins=20, filepath='model_performance/tau_label_dists.png', show=False)











