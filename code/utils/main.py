import os
import glob
import sys
from itertools import product
from collections import Iterable

import numpy as np
import pandas as pd
import scipy.stats

from .statistic import fdr, core_corr, core_corr_tensor, ranksums_vectorized, f_oneway_vectorized, pearson_pvalue, spearman_pvalue
from .function import percentage, dropnan, handle_colors, sort_custom, filter_by_quantile
from .exceptions import MethodError
from .eplot.core import scatterplot, cateplot, heatmap
from .eplot.base import savefig
from .eplot.constants import MCMAP
from .eplot.plot_func import volcanoplot


class Dataset:
    def __init__(self, dirpath=os.getcwd()):        
        self._dirpath = dirpath

        if not os.path.isdir(os.path.join(dirpath, 'document')):
            FileNotFoundError("The document file is not exist")

        self.__object_name = os.path.split(os.path.abspath(self._dirpath))[-1]
        self._data = {}
        self._valid_dfs_names = []
        self._color_map = {}
        omic_names = [
            f.split('.', -1)[0] for f in os.listdir(os.path.join(self._dirpath, 'document'))]
        self._load_dataset(omic_names)
        self._initialized = True

    def _get_df_path(self, omic_name):
        try:
            dataset_path = glob.glob(
                os.path.join(self._dirpath, 'document', omic_name + '.*'))[0]
        except:
            raise ValueError('Please check your name parameter.')

        return dataset_path

    def _get_dataframe(self, omic_name):
        omic_path = self._get_df_path(omic_name)
        if omic_path.endswith('.maf'):
            index_col = None
        elif omic_name == 'phospho':
            index_col = [0]
        elif omic_name == 'color':
            index_col = [0, 1]
        else:
            index_col = [0]

        if omic_path.endswith('.csv'):
            df = pd.read_csv(omic_path, index_col=index_col)

        elif omic_path.endswith(('.txt', 'tsv', 'maf')):
            df = pd.read_table(omic_path, index_col=index_col)

        else:
            if omic_path.endswith('.pickle'):
                df = pd.read_pickle(omic_path)
            elif omic_path.endswith('.feather'):
                df = pd.read_feather(omic_path)
            else:
                raise ValueError(
                    "dataset file type should be one of csv, tsv, txt, pickle, feather, please check your file type."
                )
            df = df.set_index(df.columns[index_col].tolist())

        df.name = omic_name
        self._data[omic_name] = df

        if not omic_name in self._valid_dfs_names:
            self._valid_dfs_names.append(omic_name)

    def _load_dataset(self, names):
        for omic_name in names:
            try:
                self._get_dataframe(omic_name)
            except:
                continue
        if 'color' in names and 'category' in self._valid_dfs_names:
            self._color_map = handle_colors(
                self._data['color'], self._data['category'])

    def _handle_group(self, file_type, group_name, part_element=None):
        group_file = self._data[file_type][group_name].dropna()
        if part_element:
            if any(set(part_element).difference(set(group_file.unique()))):
                raise ValueError('{0} with wrong elements, please check the part_element parameter'.format(part_element))
            else:
                group_file = group_file[group_file.isin(part_element)].pipe(
                    sort_custom, part_element, group_name)
        else:
            group_file = group_file.sort_values()
        return group_file

    def __getattr__(self, __name):
        if __name in self._data:
            return self._data[__name]
        else:
            return object.__getattribute__(self, __name)

    def __str__(self):
        return 'Load {0} datasets from {1} project: {2}'.format(
            len(self._valid_dfs_names), self.__object_name, ', '.join(self._valid_dfs_names))

    @staticmethod
    def delete_nan_array(df, axis=1):
        return df[df.max(axis=axis) != df.min(axis=axis)]

class Group(Dataset):
    def __init__(self,
                 dirpath=os.getcwd(),
                 group_name=None,
                 dataset_type=None,
                 file_type='category',
                 thresh=1e-5,
                 part_element=None,
                 param_method='mean',
                 statistic_method='log2',
                 ttest_kwargs = {},
                 fdr_method='i',
                 dividend=None,
                 palette={},
                 *args,
                 **kwargs):

        super().__init__(dirpath)
        self._group_name = group_name
        self._file_type = file_type
        self._dataset_type = dataset_type
        self._part_element = part_element
        self._thresh = thresh
        self._param_method = param_method
        self._statistic_method = statistic_method
        self._ttest_kwargs = ttest_kwargs
        self._fdr_method = fdr_method
        self._dividend = dividend
        self._palette = palette
        
        self._group_check_params()

    def _group_check_params(self):
        if self._group_name and self._file_type and self._dataset_type:
            if self._group_name not in self._data[self._file_type].columns:
                raise ValueError(
                    "{0} is not in {1} dataset, please check the group_name parameter."
                    .format(self._group_name, self._file_type))
            self.__group_pipeline()
        elif self._dataset_type:
            self._tmp_dataset = self._data[self._dataset_type]

    def __group_pipeline(self):
        group_file = self._handle_group(self._file_type, self._group_name, self._part_element)
        self._tmp_dataset = self._data[self._dataset_type].reindex(
            group_file.index,
            axis=1).dropna(axis=1, how='all').pipe(self.delete_nan_array).pipe(dropnan, thresh=self._thresh)

        tmp = {
            name: self._tmp_dataset.reindex(group.index,
                                            axis=1).dropna(axis=1, how='all').values
            for name, group in group_file.groupby(group_file, sort=False)
        }

        self._group_values = list(tmp.keys())
        self.__group = list(tmp.values())
        self.__group_set_params()
        
        self.__group_cal_values()
        if len(self._part_element) != 1:
            self.__table()

    def __group_set_params(self):
        if len(self._group_values) == 2:
            if self._dividend == self._group_values[0]:
                self._change = False
                self._dividend, self._divisor = self._group_values

            else:
                self._change = True
                self._divisor, self._dividend = self._group_values
        self._palette = self._color_map.get(self._group_name, MCMAP[:len(self._group_values)])
        if isinstance(self._palette, list):
            self._palette = dict(zip(self._group_values, self._palette))

        if self._part_element:
            self._palette = {k: v for k, v in self._palette.items() if k in self._part_element}
        else:
            self._part_element = self._group_values

    def __group_cal_values(self):
        mean_value = np.array(
            list(map(lambda x: np.nanmean(x, axis=1), self.__group)))
        median_value = np.array(
            list(map(lambda x: np.nanmedian(x, axis=1), self.__group)))
        std_value = np.array(
            list(map(lambda x: np.nanstd(x, axis=1, ddof=1), self.__group)))
        
        percentage_value = np.array(
            list(
                map(lambda x: percentage(x, axis=1),
                    self.__group)))
        cv_value = std_value / mean_value

        if len(self._group_values) == 1:
            inference_statistics = []

        elif len(self._group_values) == 2:
            inference_statistics = self.__two_groups_cal_statistic_prob()
        else:
            inference_statistics = self.__multi_groups_cal_statistic_prob()

        self.__param_values = dict(
            zip(['mean', 'median', 'std', 'cv', 'percentage', 'inference_statistics'], [
                mean_value, median_value, std_value, cv_value,
                percentage_value, inference_statistics
            ]))

        self.__out_index = np.any(
            self.__param_values['percentage'] > self._thresh, axis=0)

    def __dateset_preprocess_for_statistic(self):
        statistic_methods = {'log2': np.log2, 'log10': np.log10}

        if self._statistic_method in statistic_methods.keys(
        ) and self._tmp_dataset.min().min() > 0:
            statistic_data = list(
                map(
                    lambda x: statistic_methods[self._statistic_method]
                    (x), self.__group))
        else:
            statistic_data = self.__group

        return statistic_data

    def __two_groups_cal_statistic_prob(self):
        statistic_data = self.__dateset_preprocess_for_statistic()

        ttest_statistic, ttest_pvalues = np.asarray(scipy.stats.ttest_ind(*statistic_data, axis=1, equal_var=True, nan_policy='omit', **self._ttest_kwargs))
        adjust_ttest_pvalues = fdr(ttest_pvalues, self._fdr_method)[1]
        ranksums_statistic, ranksums_pvalues = ranksums_vectorized(statistic_data[0], statistic_data[1])
        adjust_ranksums_pvalues = fdr(ranksums_pvalues.copy(), self._fdr_method)[1]

        return ttest_statistic, ttest_pvalues, adjust_ttest_pvalues, ranksums_statistic, ranksums_pvalues, adjust_ranksums_pvalues

    def __multi_groups_cal_statistic_prob(self):
        statistic_data = self.__dateset_preprocess_for_statistic()

        anova_statistics, anova_pvalues = f_oneway_vectorized(*statistic_data, axis=1)
        adjust_anova_pvalues = fdr(anova_pvalues, self._fdr_method)[1]
        kruskal_statistics, kruskal_pvalues = np.asarray(scipy.stats.kruskal(*statistic_data, axis=1, nan_policy='omit'))
        adjust_kruskal_pvalues = fdr(kruskal_pvalues, self._fdr_method)[1]

        return anova_statistics, anova_pvalues, adjust_anova_pvalues, kruskal_statistics, kruskal_pvalues, adjust_kruskal_pvalues

    def __table(self):
        tmp_param_values = self.__param_values[self._param_method]
        group_values = [str(i) for i in self._group_values]
        if len(group_values) == 2:
            if any(np.hstack(tmp_param_values)<0):
                ratio = tmp_param_values[0] - tmp_param_values[1]
                nega_annot = True 
            else:
                ratio = tmp_param_values[0] / tmp_param_values[1]
                nega_annot = False 
            
            ratio_label = self._dividend + '_vs_' + self._divisor
            table_columns = [
                group_values[0] + '_' + self._param_method,
                group_values[1] + '_' + self._param_method,
                ratio_label,
                'ttest_statistics',
                'ttest_pvalues',
                'ttest_fdr',
                'ranksums_statistics',
                'ranksums_pvalues',
                'ranksums_fdr',
                group_values[0] + '_percentage',
                group_values[1] + '_percentage',
            ]
            self._group_table = pd.DataFrame(np.vstack(
                (tmp_param_values, ratio,
                 self.__param_values['inference_statistics'],
                 self.__param_values['percentage'])).transpose(),
                index=self._tmp_dataset.index,
                columns=table_columns)
            if self._change:
                if nega_annot:
                    self._group_table.iloc[:, 2] = -self._group_table.iloc[:, 2]
                else:
                    self._group_table.iloc[:, 2] = 1.0 / self._group_table.iloc[:, 2]
                self._group_table.name = '_'.join(
                    [self._dataset_type, self._group_name])

        else:
            table_columns = [
                '_'.join((label, self._param_method))
                for label in group_values
            ]
            table_columns.extend(
                ['anova_statistics', 'anova_pvalues', 'anova_fdr', 'kruskal_statistics', 'kruskal_pvalues', 'kruskal_fdr'])
            table_columns.extend([
                '_'.join((label, 'percentage'))
                for label in group_values
            ])
            self._group_table = pd.DataFrame(np.vstack(
                (tmp_param_values, self.__param_values['inference_statistics'],
                 self.__param_values['percentage'])).transpose(),
                index=self._tmp_dataset.index,
                columns=table_columns)

    @property
    def table(self):
        return self._group_table.loc[self._tmp_dataset.index[self.__out_index]]

    @property
    def param_table(self):
        param_table_columns = []
        for name in ['mean', 'median', 'standard', 'cv', 'percentage']:
            param_table_columns.extend([
                group_name + '_' + name for group_name in self._group_values
            ])

        self._param_table = pd.DataFrame(np.vstack(
            (self.__param_values['mean'], self.__param_values['median'], self.__param_values['std'], self.__param_values['cv'],
             self.__param_values['percentage'])).transpose(),
            index=self._tmp_dataset.index,
            columns=param_table_columns)

        return self._param_table

    def merge_data_group(self, data_element, group_name=None, data_type=None, part_element=None, join_method='inner', sort='element', sort_group=None, ascending=True):
        if not group_name:
            group_name = [self._group_name]
        group_name_df = self._data[self._file_type].loc[:, group_name].dropna(how='all')
        if not part_element:
            part_element = self._part_element

        group_name_df = group_name_df[group_name_df[group_name[0]].isin(part_element)].dropna()
        group_name_df = sort_custom(group_name_df, order=part_element, label=group_name[0])

        columns_name = self._tmp_dataset.index.name
        if data_type:
            data = pd.DataFrame(columns=group_name_df.index)
            for dt in data_type:
                add_df = self._data[dt].reindex(self._tmp_dataset.columns, axis=1).dropna(how='all', axis=1).reindex(data_element).dropna(how='all')
                add_df.index = add_df.index + '|' + dt
                data = pd.concat([data, add_df], join=join_method)
            if sort.startswith('el'):
                data.index = pd.MultiIndex.from_tuples(data.index.str.split('|', n=-1).tolist())
                data = data.loc[data_element]
                data.index = data.index.map('|'.join)
        else:
            data = self._tmp_dataset.reindex(data_element).dropna(how='all')

        data = pd.concat([data.T, group_name_df], axis=1, join='inner').set_index(group_name, append=True)
        data.columns.name = columns_name
        return data

class Correlation(Dataset):
    def __init__(self,
                 dirpath=os.getcwd(),
                 name1=None,
                 name2=None,
                 element1=None,
                 element2=None,
                 file_type='category',
                 group_name=None,
                 part_element=None,
                 thresh=0,
                 cal_type='other',
                 fdr_method='i',
                 fdr_type='local',
                 algorithm='all',
                 *args,
                 **kwargs):
        super().__init__(dirpath)

        self._corr_name1 = name1
        self._corr_name2 = name2
        self._corr_element1 = element1
        self._corr_element2 = element2
        self._corr_file_type = file_type
        self._corr_group_name = group_name
        self._corr_part_element = part_element
        self._corr_thresh = thresh
        self._corr_cal_type = cal_type
        self._corr_fdr_method = fdr_method
        self._corr_fdr_type = fdr_type
        self._corr_algorithm = algorithm

        self._corr_check_params()

    def _corr_check_params(self):
        if self._corr_name1 and self._corr_name2 and self._corr_element1 and self._corr_element2:
            self.__corr_pipeline()

    def __handle_omic_label(self, omic_label, corr_group_name):
        if isinstance(omic_label, str):
            if omic_label == 'all':
                return self._data[corr_group_name].index
            else:
                return [omic_label]
        elif isinstance(omic_label, Iterable):
            return omic_label
        else:
            raise ValueError('Please check your omic_label of {}'.format(corr_group_name))

    def __corr_func(self, algorithm, matrix1_value, matrix2_value):
        if algorithm == 'spearman':
            corr_name = algorithm + '_rho'
        else:
            corr_name = algorithm + '_corr'

        if all((len(self._handle_element1)>1, len(self._handle_element2)>1, not self._corr_cal_type.startswith('co'))):
            matrix1_value[matrix1_value!=matrix1_value] = 0
            matrix2_value[matrix2_value!=matrix2_value] = 0
            dof, corr = core_corr_tensor(matrix1_value, matrix2_value, method=algorithm)

        else:
            count, dof, corr = core_corr(matrix1_value, matrix2_value, method=algorithm)
            self._corr_value['count'] = self._corr_value.get('count', count)
            if self._corr_cal_type.startswith('co'):
                self._corr_value['frequence_'+self._corr_name1] = self._corr_value.get('frequence_'+self._corr_name1, count / np.count_nonzero(~np.isnan(matrix1_value), axis=1))
                self._corr_value['frequence_'+self._corr_name2] = self._corr_value.get('frequence_'+self._corr_name2, count / np.count_nonzero(~np.isnan(matrix2_value), axis=1))
            else:
                self._corr_value['frequence'] = self._corr_value.get('frequence', count / np.count_nonzero(~np.isnan(matrix1_value), axis=1))

        self._corr_value[corr_name] = corr

        if algorithm == 'spearman':
            prob = spearman_pvalue(corr, dof)
        else:
            prob = pearson_pvalue(corr, dof)

        self._corr_value[algorithm+'_pvalues'] = prob
        if self._corr_fdr_type == 'global':
            prob = prob.faltten()
        if prob.ndim == 1:
            prob = prob[None, :]
        fdr_ = np.apply_along_axis(
            fdr, 1, prob, method=self._corr_fdr_method)[:, 1][0]
        self._corr_value[algorithm+'_fdr'] = fdr_

    def __cal_cor_value(self):
        matrix1_value = self._data[self._corr_name1].loc[
            self._handle_element1, self._corr_columns].values
        matrix2_value = self._data[self._corr_name2].loc[
            self._handle_element2, self._corr_columns].values

        if 'spearman' in self._corr_algorithm.lower() or 'all' == self._corr_algorithm.lower():
            self.__corr_func('spearman', matrix1_value, matrix2_value)
        if 'pearson' in self._corr_algorithm.lower() or 'all' == self._corr_algorithm.lower():
            self.__corr_func('pearson', matrix1_value, matrix2_value)

    def __corr_pipeline(self):
        pre_element1 = self.__handle_omic_label(
            self._corr_element1, self._corr_name1)
        pre_element2 = self.__handle_omic_label(
            self._corr_element2, self._corr_name2)

        pre_omic1_dataset = self._data[self._corr_name1].loc[pre_element1].dropna(
            axis=1, how='all').pipe(self.delete_nan_array)
        pre_omic2_dataset = self._data[self._corr_name2].loc[pre_element2].dropna(
            axis=1, how='all').pipe(self.delete_nan_array)

        self._corr_columns = pre_omic1_dataset.columns.intersection(
            pre_omic2_dataset.columns)

        pre_omic1_dataset = dropnan(pre_omic1_dataset[self._corr_columns], thresh=self._corr_thresh)
        pre_omic2_dataset = dropnan(pre_omic2_dataset[self._corr_columns], thresh=self._corr_thresh)
        
        if self._corr_group_name and self._corr_file_type:
            omic_group_name_id = self._handle_group(self._corr_file_type, self._corr_group_name, self._corr_part_element).index
            self._corr_columns = omic_group_name_id.intersection(
                self._corr_columns)

        self._handle_element1 = pre_omic1_dataset.index
        self._handle_element2 = pre_omic2_dataset.index

        if min(len(self._handle_element1), len(
                self._handle_element2)) > 1:
            if self._corr_cal_type.startswith('co'):
                self._handle_element1 = self._handle_element2 = self._handle_element1.intersection(self._handle_element2)
                if len(self._handle_element1) == 0:
                    raise ValueError('{} and {} has not overlapped index'.format(
                        self._corr_name1, self._corr_name2))

        elif len(self._handle_element2) < len(self._handle_element1):
            self._handle_element1, self._handle_element2 = self._handle_element2, self._handle_element1
            self._corr_name1, self._corr_name2 = self._corr_name2, self._corr_name1

        self._corr_value = {}
        self.__cal_cor_value()
        self.__corr_table()

    def __corr_table(self):
        if not ((len(self._handle_element1) == 1
                 or len(self._handle_element2) == 1)
                or self._corr_fdr_type.lower() == 'local'):
            raise MethodError(
                'corr_table only suitable for one vs. n data type or n vs. n data when fdr_type is local , please consider spearman_rho, spearman_prob, spearman_fdr, pearson_corr, pearson_prob and pearsonfdr function to obtain correlation, probability and FDR matrix seperately.'
            )
        else:
            self._corr_table = pd.DataFrame(np.vstack(
                list(self._corr_value.values())).T,
                index=self._handle_element2,
                columns=self._corr_value.keys())
            return self._corr_table
    
    @property
    def corr_table(self):
        return self._corr_table


class Analysis(Group, Correlation):
    def __init__(self,
                 dirpath=os.getcwd(),
                 group_name=None,
                 dataset_type=None,
                 file_type='category',
                 thresh=1e-5,
                 part_element=None,
                 param_method='mean',
                 statistic_method='log2',
                 ttest_kwargs = {},
                 fdr_method='i',
                 dividend=None,
                 palette={},
                 name1=None,
                 name2=None,
                 element1=None,
                 element2='all',
                 cal_type='other',
                 fdr_type='local',
                 algorithm='all',
    ):

        super(Analysis,
              self).__init__(dirpath, group_name, dataset_type, file_type, thresh, part_element, param_method, statistic_method, ttest_kwargs, fdr_method, dividend, palette, name1, name2, element1, element2, cal_type, fdr_type, algorithm, type, thresh)

        self._corr_param_collections = {
            'name1': self._corr_name1,
            'name2': self._corr_name2,
            'element1': self._corr_element1,
            'element2': self._corr_element2,
            'cal_type': self._corr_cal_type,
            'fdr_method': self._corr_fdr_method,
            'fdr_type': self._corr_fdr_type,
            'algorithm': self._corr_algorithm,
            'file_type': self._corr_file_type,
            'group_name': self._corr_group_name,
            'part_element': self._corr_part_element,
            'thresh': self._corr_thresh
        }
        self._group_param_collections = {
            'group_name': self._group_name,
            'dataset_type': self._dataset_type,
            'file_type': self._file_type,
            'thresh': self._thresh,
            'part_element': self._part_element,
            'param_method': self._param_method,
            'statistic_method': self._statistic_method,
            'ttest_kwargs': self._ttest_kwargs,
            'fdr_method': self._fdr_method,
            'dividend': self._dividend
        }
        self._tmp_dict = {
            'group': self._group_param_collections,
            'corr': self._corr_param_collections
        }

    def __set_default_params(self, params=[]):
        for param in params:
            setattr(self, '_'+param, None)

    def set_param(self, function_name, **kwargs):
        if function_name.lower() not in self._tmp_dict.keys():
            raise ValueError(
                'The first param of set_param function should be one of {0}, please check it.'
                .format(', '.join(self._tmp_dict.keys())))

        else:
            error_param = list(
                set(kwargs.keys()) - set(self._tmp_dict[function_name].keys())
            )
            if error_param:
                raise ValueError(
                    'Please check the input parameter name: {0}'.format(
                        ', '.join(error_param)))
            if function_name == 'group':
                params = []
                if 'group_name' in kwargs and 'group_name' != self._group_name:
                    params.extend(['dividend', 'part_element', 'palette'])
                if 'file_type' in kwargs and 'file_type' != self._file_type:
                    params.extend('group_name')
                self.__set_default_params(params)

                for k, v in kwargs.items():
                    if k == 'thresh':
                        v = {0: 1e-5, 1: 1-1e-5}.get(v, v)

                    self.__dict__['_' + k] = v
                    self._tmp_dict[function_name][k] = v
                self._group_check_params()

            elif function_name == 'corr':
                self._corr_value = {}
                for k, v in kwargs.items():
                    self.__dict__['_corr_' + k] = v
                    self._tmp_dict[function_name][k] = v
                self._corr_check_params()

    def get_param(self):
        return pd.DataFrame().from_dict(self._tmp_dict).rename(columns={
            0: 'Value'
        }).rename_axis('Parameter', axis='index')

    def range(self,
              axis='columns',
              method='mean',
              c=['grey'],
              s=5,
              highlight_annots={},
              figsize=(4, 2.5),
              ylabel='$\mathregular{Log_{10}}$(FoT)',
              title=None,
              labels_hide=['x'],
              save=None,
              outname=None,
              **kwargs):
        if method not in ['mean', 'median']:
            raise ValueError(
                'rangeplot function only support mean and median parameter now.')
        else:
            pass

        if self._group_name:
            min_value = np.log10(self._data[self._dataset_type].min().min())
            plotdata = pd.DataFrame(np.log10(self._Group__param_values[method].T), columns=self._group_values, index=self._tmp_dataset.index).stack().astype('float32').rename_axis(['Symbol', self._group_name]).groupby(
                self._group_name).apply(lambda x: x.sort_values(ascending=False).rename(ylabel).reset_index()).droplevel(0).set_index('Symbol', append=True).droplevel(0).replace(min_value, np.nan).dropna()
            plotdata = plotdata.groupby('cohort').apply(lambda x: x.reset_index().rename_axis('rank').reset_index().set_index(plotdata.index.name))
            plotdata['rank'] = plotdata['rank'] + 1
        else:
            stand_method = {'mean': np.mean,
                            'median': np.median}.get(method, None)
            plotdata = np.log10(stand_method(self._data[self._dataset_type], axis=axis)).sort_values(
                ascending=False).rename(ylabel).to_frame()
            plotdata.insert(0, 'rank', range(1, len(plotdata)+1))

        palette = self._palette.copy()

        if highlight_annots:
            for k, v in highlight_annots.items():
                highlight_points, color, size = v
                plotdata.loc[highlight_points, ('hue', 'size')] = (k, size)
                palette[k] = color
            plotdata.loc[:, 'hue'] = plotdata.loc[:, 'hue'].fillna('others')
            plotdata.loc[:, 'size'] = plotdata.loc[:, 'size'].fillna(s)
            palette['others'] = c[0]
            plotdata = plotdata.iloc[:, [0, 2, 3, 4]]
        else:
            highlight_points = []

        ax = scatterplot(plotdata,
                         figsize=figsize,
                         palette=palette,
                         title=title,
                         labels_hide=labels_hide,
                         highlight_points=highlight_points,
                         **kwargs)
        if save:
            if title and not outname:
                outname = title
            else:
                pass
            savefig(ax, outpath=os.path.join(self._dirpath, 'figure'), outname=outname, plot_type=sys._getframe(
            ).f_code.co_name, out_format=save)

        return ax

    def cate(self, elements, data_type=None, data_type_annot='auto', method='ranksums', value_log_transform=None, quantile=False, ax=None, figsize=(1.8, 1.8), one_plot=False, category_type=['violin', 'strip'], orient='v', title=None, ticklabels_format=['y'], one_pdf=False, save=False, outname=None, **kwargs):
        if isinstance(elements, str):
            elements = [elements]
        if not data_type:
            data_type = [self._dataset_type]
        data = self.merge_data_group(elements, data_type=data_type).dropna(how='all', axis=1)
        
        if (data_type_annot == 'no') or (data_type_annot == 'auto' and len(data_type) == 1):
            data = data.rename(columns=lambda x: x.split('|')[0])
            elements_loop = elements
        else:
            elements_loop = list(['|'.join(i) for i in product(elements, data_type)])
        
        if len(elements_loop) == 1:
            order = self._part_element
            name = elements[0]
        elif not one_plot:
            order = self._part_element
            name = ''
        else:
            order = elements_loop
            name = ' '      
        data = data.stack().swaplevel(1, 2).rename(name).astype(float)
        if quantile:
            data = data.groupby(['term', self._group_name]).apply(lambda x: filter_by_quantile(x)).droplevel([0, 1])

        if not title:
            title = name
        
        _, _, hue = data.index.names

        if value_log_transform:
            assert value_log_transform in ['log2', 'log10'], "parameter value_log_transform should be one of 'log2' and 'log10'"
            data = {'log2': np.log2, 'log10': np.log10}.get(value_log_transform)(data)
            kwargs['log_transform'] = 'no'

        params = {'hue_order': self._part_element, 'method': method, 'category_type': category_type, 'palette': self._palette, 'ax': ax, 'figsize': figsize, 'title': title, 'orient': orient, 'ticklabels_format': ticklabels_format}

        if one_plot:
            axs = cateplot(data, hue=hue, order=order, **params, **kwargs)
        else:
            axs = []
            for element in elements_loop:                
                tmp_data = data.xs(element, axis=0, level=1).rename(element)
                params['title'] = element
                axs.append(cateplot(tmp_data, order=order,
                          hue=hue, **params, **kwargs))

        if save:
            if title and not outname:
                outname = title
            else:
                pass
            savefig(ax, outpath=os.path.join(self._dirpath, 'figure'), outname=outname, plot_type=sys._getframe(
            ).f_code.co_name, out_format=save)
        return axs

    def heat(self, elements, annot_dict=None, lut=None, group_name=None, data_type=None, data_type_annot='auto', join_method='inner', sort='element', sort_group=None, z_score=0, **kwargs):
        if not data_type:
            data_type = [self._dataset_type]
        if not group_name:
            group_name = [self._group_name]
        
        plotdata = self.merge_data_group(elements, group_name=group_name, data_type=data_type, join_method=join_method, sort=sort, sort_group=sort_group).astype(float)

        if annot_dict:
            tmp_columns = plotdata.columns.str.split('|', expand=True).get_level_values(0)
            plotdata.columns = pd.MultiIndex.from_arrays(np.vstack((tmp_columns, [tmp_columns.map(v) for k, v in annot_dict.items()])), names=np.hstack(('Genes', list(annot_dict.keys()))))

        remove = False
        if (data_type_annot == 'no') or (data_type_annot == 'auto' and len(data_type) == 1):
                remove = True
        if remove:
            plotdata = plotdata.rename(columns=lambda x: x.split('|')[0], level=0)

        if len(group_name) > 1:
            plotdata_index_element = dict(zip(plotdata.index.names[1:], [i.values for i in plotdata.droplevel(0).index.levels]))
            lut_default = {i: {k: v for k, v in j.items() if k in plotdata_index_element[i]} for i, j in self._color_map.items() if i in group_name}
        else:
            lut_default = {i: {k: v for k, v in j.items() if k in self._part_element} for i, j in self._color_map.items() if i in group_name}

        if lut:
            lut_default.update(lut)

        if 'col_cluster' in kwargs.keys() or 'row_cluster' in kwargs.keys():
            plotdata = plotdata.fillna(1e-5)
        
        if any([kwargs.get('col_cluster', None), kwargs.get('row_cluster', None)]):
                plotdata = plotdata.fillna(0)

        ax = heatmap(plotdata.T, lut=lut_default, z_score=z_score, **kwargs)
        return ax

    def scatter(self, elements, volcano=False, sig_log_transform=True, hue=None, size=None, highlight_points=None, palette=None, ax=None, figsize=(1.8, 1.8), title=None, adjust_axes=True, ticklabels_hide=[], ticklabels_format=['y'], ticklabels_wrap=[], wrap_length=None, spines_hide=[], labels_hide=[], **kwargs):
        series_list = []
        for table_name in ['_group_table', '_corr_table', '_reg_table']:
            if hasattr(self, table_name):
                add_new = [self.__dict__[table_name][element] for element in elements if element in self.__dict__[table_name].columns]
                series_list.extend(add_new)
        table = pd.concat(series_list, axis=1, join='inner').reindex(elements, axis=1).dropna(how='all', axis=1)
        if len(elements) == 3:
            table = table.iloc[:, [0, 2, 1]]

        columns_shape = table.columns.shape[0]
        if table.columns.nunique() != columns_shape:
            if columns_shape > 2:
                rename_column = table.columns[2]
                table = pd.concat([table.iloc[:, :2], table.iloc[:, 2:].rename(columns={rename_column: rename_column+'_hue'})], axis=1)
            if columns_shape > 3:
                rename_column = table.columns[3]
                table = pd.concat([table.iloc[:, :3], table.iloc[:, 3:].rename(columns={rename_column: rename_column+'_size'})], axis=1)

        if volcano:
            sig_up_color, sig_down_color = self._palette[self._dividend], self._palette[self._divisor]
            if not title:
                title = '{} vs. {}'.format(self._dividend, self._divisor)
            if isinstance(highlight_points, dict):
                highlight_points = np.unique(np.hstack(self.get_gene_from_enrichment(highlight_points).values())).tolist()

            ax = volcanoplot(table, title=title, sig_up_color=sig_up_color, sig_down_color=sig_down_color, highlight_points=highlight_points, adjust_axes=adjust_axes, ticklabels_hide=ticklabels_hide, ticklabels_format=ticklabels_format, ticklabels_wrap=ticklabels_wrap, wrap_length=wrap_length, spines_hide=spines_hide, labels_hide=labels_hide, ax=ax, figsize=figsize, **kwargs)
        else:
            if sig_log_transform:
                if not isinstance(sig_log_transform, Iterable):
                    sig_log_transform = table.filter(regex="pvalues|fdr").columns
                table[sig_log_transform] = -np.log10(table[sig_log_transform])
                def rename_columns(x):
                    if x in sig_log_transform:
                        x = '-Log10({})'.format(x)
                    return x.replace('_', ' ').capitalize()

                table = table.rename(columns=lambda x: rename_columns(x))
            if hue:
                if isinstance(hue, (pd.Series)):
                    hue = hue.to_frame()

                if isinstance(hue, (pd.DataFrame)):
                    hue = hue.to_dict()

                if isinstance(hue, dict):
                    if not any(isinstance(i, dict) for i in hue.values()):
                        hue = {'hue': hue}
                    k, v = list(hue.items())[0]
                    v = self.get_gene_from_enrichment(v)

                for k1, v1 in v.items():
                    gene = np.intersect1d(table.index, v1)
                    table.loc[gene, k] = k1
                table[k] = table[k].fillna('')
                if size:
                    if size == 'hue':
                        table.loc[:, 'size'] = table[k].apply(lambda x: 'annot' if x!='' else x)
                    else:
                        if any(isinstance(i, dict) for i in size.values()):
                            k, v = list(size.items())[0]
                        else:
                            k = 'size'
                            v = size
                        table.loc[:, k] = table.index.map(v).fillna('')
                else:
                    table = table.iloc[:, [0, 2, 1]]
                table = table.sort_values(k, ascending=True)
            
            if highlight_points == 'hue':
                highlight_points = np.hstack(list(v.values())).tolist()
            
            ax = scatterplot(table, title=title, palette=palette, highlight_points=highlight_points, adjust_axes=adjust_axes, ticklabels_hide=ticklabels_hide, ticklabels_format=ticklabels_format, ticklabels_wrap=ticklabels_wrap, wrap_length=wrap_length, spines_hide=spines_hide, labels_hide=labels_hide, ax=ax, figsize=figsize, **kwargs)
        return ax