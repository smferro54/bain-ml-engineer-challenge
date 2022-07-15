#We take the datasets and replicate the transformations the data scientist made. 

import pandas as pd
import numpy as np
import locale
locale.setlocale(locale.LC_TIME, 'es_US.UTF-8')
from datetime import date
import argparse
import logging
logging.basicConfig(level = logging.INFO)
import mlflow

def _convert_int(x):
    """
    This function replaces the dot, which is used as a thousand separator in some series and as a visual aid in others. 
    """
    return int(x.replace('.', ''))

def _to_100(x):
    """
    Imacec data comes separated by a dot every 3 positions for visual aid. However, the original number exists in the 85-120 range. 
    """
    x = x.split('.')
    if x[0].startswith('1'): #es 100+
        if len(x[0]) >2:
            return float(x[0] + '.' + x[1])
        else:
            x = x[0] + x[1]
            return float(x[0:3] + '.' + x[3:])
    else:
        if len(x[0])>2:
            return float(x[0][0:2] + '.' + x[0][-1])
        else:
            x = x[0] + x[1]
            return float(x[0:2] + '.' + x[2:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSV file - precipitaciones")
    #Check encodings
    try:
        parser.add_argument('csv_precipitaciones', type=argparse.FileType('r', encoding='ascii', errors='strict'))
    except ValueError:
        raise Exception("The encoding of precipitaciones is not ascii")
    
    try:
        parser.add_argument('csv_banco_central', type=argparse.FileType('r', encoding='ascii', errors='strict'))
    except ValueError:
        raise Exception("The encoding of data banco central is not ascii")

    try:
        parser.add_argument('csv_precio_leche', type=argparse.FileType('r', encoding='ascii', errors='strict'))
    except ValueError:
        raise Exception("The encoding of data precio leche is not ascii")
    
    args = parser.parse_args()
    
    #Preprocessing precipitaciones
    precipitaciones = pd.read_csv(args.csv_precipitaciones)
    precipitaciones['date'] = pd.to_datetime(precipitaciones['date'], format = '%Y-%m-%d')
    precipitaciones = precipitaciones.sort_values(by = 'date', ascending = True).reset_index(drop = True)
    logging.info(f'Length of precipitaciones load: {len(precipitaciones)}')
    precipitaciones.dropna().reset_index(drop = True, inplace=True)
    logging.info(f'Length of precipitaciones after dropping nan: {len(precipitaciones)}')
    precipitaciones.drop_duplicates(subset='date').reset_index(drop = True, inplace=True)
    logging.info(f'Length of precipitaciones after dropping dupplicates: {len(precipitaciones)}')

    #Preprocessing banco central
    banco_central = pd.read_csv(args.csv_banco_central)
    banco_central['Periodo'] = banco_central['Periodo'].apply(lambda x: x[0:10])
    banco_central['Periodo'] = pd.to_datetime(banco_central['Periodo'], format = '%Y-%m-%d', errors = 'coerce')
    logging.info(f'Length of banco_central load: {len(banco_central)}')
    banco_central.dropna(subset=["Periodo"]).reset_index(drop = True, inplace=True)
    logging.info(f'Length of banco_central after dropping nan: {len(banco_central)}')
    banco_central.drop_duplicates(subset = 'Periodo', inplace = True)
    logging.info(f'Length of banco_central after dropping dupplicates: {len(precipitaciones)}')
    
    cols_pib = [x for x in list(banco_central.columns) if 'PIB' in x] 
    cols_pib.extend(['Periodo'])
    banco_central_pib = banco_central[cols_pib]
    banco_central_pib = banco_central_pib.dropna(how = 'any', axis = 0)

    for col in cols_pib:
        if col == 'Periodo':
            continue
        else:
            banco_central_pib[col] = banco_central_pib[col].apply(lambda x: _convert_int(x))
    logging.info(f'Length of banco_central_pib: {len(banco_central_pib)}')

    cols_imacec = [x for x in list(banco_central.columns) if 'Imacec' in x]
    cols_imacec.extend(['Periodo'])
    banco_central_imacec = banco_central[cols_imacec]
    banco_central_imacec = banco_central_imacec.dropna(how = 'any', axis = 0)

    for col in cols_imacec:
        if col == 'Periodo':
            continue
        else:
            banco_central_imacec[col] = banco_central_imacec[col].apply(lambda x: _to_100(x)) ###SF: Transform - Training, uses built-in functions
            assert(banco_central_imacec[col].max()<200) ###SF: Transform - Training, uses built-in functions. See note on IMACEC on _to_100 function.
            assert(banco_central_imacec[col].min()>30) ###SF: Transform - Training, uses built-in functions. See note on IMACEC.
    banco_central_imacec.sort_values(by = 'Periodo', ascending = True)

    banco_central_iv = banco_central[['Indice_de_ventas_comercio_real_no_durables_IVCM', 'Periodo']]
    banco_central_iv = banco_central_iv.dropna() 
    banco_central_iv = banco_central_iv.sort_values(by = 'Periodo', ascending = True) 
    banco_central_iv['num'] = banco_central_iv.Indice_de_ventas_comercio_real_no_durables_IVCM.apply(lambda x: _to_100(x))###SF: Transform - Training. This is another index, see note on IMACEC.
    banco_central_num = pd.merge(banco_central_pib, banco_central_imacec, on = 'Periodo', how = 'inner') 
    logging.info(f'Length of banco_central_num, first merge: {len(banco_central_num)}')
    banco_central_num = pd.merge(banco_central_num, banco_central_iv, on = 'Periodo', how = 'inner') 
    logging.info(f'Length of banco_central_num, second merge: {len(banco_central_num)}')

    precio_leche = pd.read_csv(args.csv_precio_leche)
    precio_leche.rename(columns = {'Anio': 'ano', 'Mes': 'mes_pal'}, inplace = True)
    precio_leche['mes'] = pd.to_datetime(precio_leche['mes_pal'], format = '%b')
    precio_leche['mes'] = precio_leche['mes'].apply(lambda x: x.month)
    precio_leche['mes-ano'] = precio_leche.apply(lambda x: f'{x.mes}-{x.ano}', axis = 1)

    precipitaciones['mes'] = precipitaciones.date.apply(lambda x: x.month)
    precipitaciones['ano'] = precipitaciones.date.apply(lambda x: x.year)
    precio_leche_pp = pd.merge(precio_leche, precipitaciones, on = ['mes', 'ano'], how = 'inner')
    precio_leche_pp.drop('date', axis = 1, inplace = True)
    logging.info(f'Length of precio_leche_pp: {len(precio_leche_pp)}')

    banco_central_num['mes'] = banco_central_num['Periodo'].apply(lambda x: x.month)
    banco_central_num['ano'] = banco_central_num['Periodo'].apply(lambda x: x.year)
    precio_leche_pp_pib = pd.merge(precio_leche_pp, banco_central_num, on = ['mes', 'ano'], how = 'inner')
    precio_leche_pp_pib.drop(['Periodo', 'Indice_de_ventas_comercio_real_no_durables_IVCM', 'mes-ano', 'mes_pal'], axis =1, inplace = True)
    logging.info(f'Length of precio_leche_pp: {len(precio_leche_pp_pib)}')

    X = precio_leche_pp_pib.drop(['Precio_leche'], axis = 1)
    y = precio_leche_pp_pib['Precio_leche']
    mlflow.log_metric("mean of dependent variable", y.mean())
    mlflow.log_metric("standard deviation of dependent variable", y.std())

    X.to_csv('X.csv', index = False)
    y.to_csv('y.csv', index = False)
    mlflow.log_artifact('X.csv')
    mlflow.log_artifact('y.csv')
