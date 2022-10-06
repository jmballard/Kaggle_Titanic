

#%% Import

import os
import pandas as pd
import numpy as np
from functools import reduce

output_submissions = os.path.join(
    "outputs",
    "submissions")

output_glm = pd.read_csv(os.path.join(output_submissions,
                           'submission_GLM.csv'))
output_glm.rename(columns = {'Survived' : 'Survived_GLM'},
                  inplace = True)

output_lgb = pd.read_csv(os.path.join(output_submissions,
                           'submission_lightgbm_cv.csv'))
output_lgb.rename(columns = {'Survived' : 'Survived_LGB'},
                  inplace = True)

output_xgb = pd.read_csv(os.path.join(output_submissions,
                           'submission_XGBoost.csv'))
output_xgb.rename(columns = {'Survived' : 'Survived_XGB'},
                  inplace = True)

output_ebm = pd.read_csv(os.path.join(output_submissions,
                           'submission_EBM.csv'))
output_ebm.rename(columns = {'Survived' : 'Survived_EBM'},
                  inplace = True)

# %% Join

final = reduce(lambda df1, df2: df1.merge(df2,
                                           on = 'PassengerId',
                                           how = 'inner'),
               [output_glm,
                  output_lgb,
                  output_xgb,
                  output_ebm])



#%% Write mixes

final['mix_1'] = np.round(np.mean(final[['Survived_GLM',
                                        'Survived_LGB',
                                        'Survived_XGB',
                                        'Survived_EBM']],
                         axis = 1)+0.00001).astype(int)

final['mix_2'] = np.round(np.mean(final[['Survived_XGB',
                                        'Survived_EBM']],
                         axis = 1)+0.00001).astype(int)


final['mix_3'] = np.round(np.mean(final[['Survived_LGB',
                                        'Survived_XGB',
                                        'Survived_XGB',
                                        'Survived_EBM']],
                         axis = 1)+0.00001).astype(int)


for mixes in range(1,4):
    
    final[['PassengerId',f'mix_{mixes}']].\
        rename(columns = {f'mix_{mixes}' : 'Survived'}).\
        to_csv(os.path.join(output_submissions,
                               f'submission_mix{mixes}.csv'),
                  index=False)