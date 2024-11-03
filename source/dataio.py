import warnings
warnings.filterwarnings('ignore')
import os
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def readfile(file_path, delimiter=None):
    # 根据文件类型选择读取方法
    if file_path.endswith('.h5'):
        return pd.read_hdf(file_path,key='data',mode='r')
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.dat'):
        return np.loadtxt(file_path, delimiter=delimiter or ' ')
    elif file_path.endswith('.fits'):
        return Table.read(file_path).to_pandas()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def savefile(data, file_path, delimiter=None):
    # 根据数据类型选择保存方法
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        if file_path.endswith('.h5'):
            df = pd.DataFrame(data)
            df.to_hdf(file_path, key='data', mode='w', format='table')
        elif file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif file_path.endswith('.fits'):
            df = pd.DataFrame(data)
            table = Table.from_pandas(df)
            table.write(file_path, format='fits', overwrite=True)
        elif file_path.endswith('.dat'):
            if isinstance(data, pd.DataFrame):
                np.savetxt(file_path, data.values, delimiter=delimiter)
            elif isinstance(data, pd.Series):
                np.savetxt(file_path, data.to_numpy(), delimiter=delimiter)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    elif isinstance(data, np.ndarray):
        if file_path.endswith('.h5'):
            df = pd.DataFrame(data)
            df.to_hdf(file_path, key='data', mode='w', format='table')
        elif file_path.endswith('.dat'):
            # 将 NumPy 数组保存为空格分隔的文本文件
            np.savetxt(file_path, data, delimiter='\t')
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    else:
        raise ValueError("Unsupported data type")
        