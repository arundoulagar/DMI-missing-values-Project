###### Import Libraries #########
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import glob
import ntpath
import os

###### Create Rsult Directories #######
try:
    os.makedirs("Imputed_Datasets")
    print("Directory ", "Imputed_Datasets", " Created ")
except FileExistsError:
    print("Directory ", "Imputed_Datasets", " already exists")
try:
    os.makedirs("NRMS_AE")
    print("Directory ", "NRMS_AE", " Created ")
except FileExistsError:
    print("Directory ", "NRMS_AE", " already exists")

#### Input Folder Name to Impute Missing Values  (such as Iris, TTTTEG, HOW ,..) #####
Folder_Name = input("Input Folder Name to impute (ex:Iris) : ")

input_folder_dir = "C:/Users/arund/Downloads/Course Project Datasets/Course Project Datasets/Incomplete Datasets Without Labels/" + Folder_Name + "/"
Original_folder_dir = "C:\\Users\\arund\\Downloads\\Course Project Datasets\\Course Project Datasets\\Original Datasets Without Labels\\"


#####   Functions Definition  ########
def read_input_file(file_path):
    Df = pd.read_excel(file_path, header=None)
    ### set column names
    col_names = []
    for col in range(Df.shape[1]):
        col_names.append('col_' + str(col + 1))
    Df.columns = col_names
    return Df


def check_numerical_categorical(df):
    A_num = []
    A_cat = []
    for col_name in df.columns:
        if (df[col_name].dtype == np.float64 or df[col_name].dtype == np.int64):
            A_num.append(col_name)
            # print(col_name," : ",df[col_name].dtype)
        if df[col_name].dtype == 'object':
            A_cat.append(col_name)
            # print(col_name," : ",df[col_name].dtype)
    return A_num, A_cat


def get_Dc_Di(Df):
    Dc_idx = []
    Di_idx = []
    for idx in range(len(Df)):
        if Df.loc[idx].isnull().sum() > 0:
            Di_idx.append(idx)
        else:
            Dc_idx.append(idx)
    Di = Df.loc[Di_idx]
    Dc = Df.loc[Dc_idx]
    return Dc, Di


#def save_estimated_file(df, result_folder_dir, result_file_name):
 #   df.to_excel(result_folder_dir + result_file_name)
 #   print("\n ###  Successfully saved as : ", result_folder_dir + result_file_name)



def get_AE(x_origin, x_estimate):
    row = x_origin.shape[0]
    col = x_origin.shape[1]
    n = row * col
    # print(n)
    m = 0
    if n == 0:
        return None
    else:
        for idx in range(len(x_origin)):
            for c in x_origin.columns:
                if x_origin[c][idx] == x_estimate[c][idx]:
                    m += 1
        return m / n


def get_nrms(x_origin, x_estimate):
    n = 0
    m = 0
    if x_origin.empty:
        return None
    elif len(x_origin) > 0 and len(x_estimate) > 0:
        for idx in range(len(x_origin)):
            for c in x_origin.columns:
                n += abs(x_origin[c][idx]) ** 2
                m += abs(x_origin[c][idx] - x_estimate[c][idx]) ** 2
        return np.sqrt(m) / np.sqrt(n)


def get_labels(Df, A_c):
    symbols = []
    for col in A_c:
        for sym in Df[col].unique().tolist():
            if sym not in symbols:
                symbols.append(sym)

    lab = {}
    i = 0
    for sym in symbols:
        if pd.isnull(sym) == False:
            lab[sym] = i
            i += 1
    inv_lab = {v: k for k, v in lab.items()}
    return lab, inv_lab


def convert_cat_numerical(Df, labels):
    for idx in Df.index:
        # print(idx)
        for col in A_c:
            # print(col)
            # print(Di[col][idx])
            if pd.isnull(Df[col][idx]) == False:
                Df[col][idx] = labels[Df[col][idx]]
    return Df


############################################
####   Start Applying DMI Algorithm  ######
############################################
file_names = []
nrms_list = []
ae_list = []

D = read_input_file(Original_folder_dir + Folder_Name + ".xlsx")
all_files_path = glob.glob(input_folder_dir + '*.xlsx')

for path in all_files_path:
    name = ntpath.basename(path).split(".")[0]
    file_names.append(name)
    print("\n ## Input File Name : ## \n\n", name)

    Df = read_input_file(path)

    print("\n\tSplit Numerical and Categorical Columns ... ")
    A_n, A_c = check_numerical_categorical(Df)
    print("\n\tA_num,A_cate : ", A_n, A_c)

    labels, inverse_labels = get_labels(Df, A_c)
    print("\n\tlabes,inv_labels : ", labels, inverse_labels)

    Df = convert_cat_numerical(Df, labels)

    Dc, Di = get_Dc_Di(Df)
    Dc_1 = Dc
    A_i = []
    for col in Di.columns:
        if Di[col].isnull().sum() > 0:
            A_i.append(col)
    print("\n\tColumn Names that have missing values : ", A_i)

    print("\n\tBuild DT for every Ai and Find Di that falls in DT_i ... ")
    Si_list = []
    Si_Di_list = []
    Dc_leaves = []
    Di_leaves = []

    for col_num in range(len(A_i)):
        target_col = A_i[col_num]
        if target_col in A_n:
            N = int(np.sqrt(len(Dc_1[target_col].unique())))
            class_labels = ["Class" + str(i) for i in range(N)]
            try:
                y = pd.cut(Dc_1[target_col], N, labels=class_labels)
            except:
                y = Dc_1[target_col]
                y = y.astype(str)
        else:
            y = Dc_1[target_col]
            y = y.astype('int')
        Xc = Dc_1.drop(target_col, axis=1)
        Xi = Di.drop(target_col, axis=1)

        T_i = DecisionTreeClassifier(max_depth=2, random_state=0)
        T_i.fit(Xc, y)

        Xc_val = Xc.values.astype("float32")
        Dc_leaf_ids = T_i.tree_.apply(Xc_val)
        Dc_leaf_ids = np.array(Dc_leaf_ids)
        Dc_id_unique, Dc_id_counts = np.unique(Dc_leaf_ids, return_counts=True)

        Dc_idx = Xc.index.tolist()
        S_i = len(Dc_id_unique)
        Si_list.append(S_i)
        Ti_leaves = [[] for _ in range(S_i)]

        for k in range(S_i):
            for idx in range(len(Dc_idx)):
                if Dc_leaf_ids[idx] == Dc_id_unique[k]:
                    Ti_leaves[k].append(Dc_idx[idx])
            Dc_leaves.append(Ti_leaves[k])

        Xi_val = Xi.values.astype("float32")
        Di_leaf_ids = T_i.tree_.apply(Xi_val)
        Di_leaf_ids = np.array(Di_leaf_ids)
        Di_id_unique, Di_id_counts = np.unique(Di_leaf_ids, return_counts=True)

        Di_idx = Xi.index.tolist()
        Di_Si = len(Di_id_unique)
        Si_Di_list.append(Di_Si)
        temp_leaves = [[] for _ in range(Di_Si)]

        for k in range(Di_Si):
            for idx in range(len(Di_idx)):
                if Di_leaf_ids[idx] == Di_id_unique[k]:
                    temp_leaves[k].append(Di_idx[idx])
            Di_leaves.append(temp_leaves[k])
    print("\n\tImpute Missing Values of Di with Dj Values ... ")
    for col_num in range(len(A_i)):
        target_col = A_i[col_num]
        s = 0
        r = 0
        for n in range(col_num):
            s += Si_list[n]
            r += Si_Di_list[n]
        for k in range(Si_Di_list[col_num]):
            Dj = Dc.loc[Dc_leaves[s + k]]
            if target_col in A_n:
                m_j = Dj[target_col].mean()
            elif target_col in A_c:
                m_j = int(Dj[target_col].mode().tolist()[0])
            Di_i = Di.loc[Di_leaves[r + k]]
            Di[target_col] = Di[target_col].replace(Di.loc[Di_i[Di_i[target_col].isnull()].index.tolist()][target_col],
                                                    m_j)

    estimate = Dc.append(Di)
    estimate = estimate.sort_index()
    for idx in estimate.index:
        for col in A_c:
            estimate[col][idx] = inverse_labels[int(estimate[col][idx])]
    ### save imputed files in Imputed_Datasets folder ###
    result_folder_name = "Imputed_Datasets/"
    result_file_name = "Imputed_" + name + '.xlsx'
    estimate.to_excel(result_folder_name + result_file_name)

    D_n = D[A_n]
    D_c = D[A_c]

    est_n = estimate[A_n]
    est_c = estimate[A_c]
    score_nrms = get_nrms(D_n, est_n)
    score_ae = get_AE(D_c, est_c)
    print("\n\tNRMS(original,imputed)  = ", score_nrms)
    print("\n\tAE(original,imputed)  = ", score_ae)
    nrms_list.append(score_nrms)
    ae_list.append(score_ae)
    df = pd.DataFrame(list(zip(file_names, nrms_list, ae_list)), columns=['File_Name', 'NRMS', 'AE'])
    df.to_excel("NRMS_AE/NRMS_AE_" + Folder_Name + ".xlsx")
#####   Save Result Scores as a Table  ######
df = pd.DataFrame(list(zip(file_names, nrms_list, ae_list)), columns=['File_Name', 'NRMS', 'AE'])
df.to_excel("NRMS_AE/NRMS_AE_" + Folder_Name + ".xlsx")
print("\n ### Successfully Saved NRMS as a file ### \n")
