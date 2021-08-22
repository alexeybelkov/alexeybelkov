from time_series_module import ForecastModel
import numpy as np
import matplotlib.pyplot as plt
import time_series_module as tsm
from functools import reduce
class CrossValid:

    def __init__(self,train_size, test_size, min_period = 0):
        self.train_size = train_size
        self.test_size = test_size
        self.min_period = min_period
        self.n_splits = None

    def split(self, data_len, step = 1, gap = 0):
        train_begin, train_end = self.min_period + 0, self.min_period + self.train_size
        test_begin, test_end = self.min_period + self.train_size + gap, self.min_period + self.train_size + gap + self.test_size
        self.n_splits = int((data_len - self.min_period - gap - self.train_size - self.test_size)/step) + 1
        while test_end <= data_len:
            yield list(range(train_begin, train_end)), list(range(test_begin, test_end))
            train_begin += step
            train_end += step
            test_begin += step
            test_end += step

def cross_valid(forecast_model, cv_dict, model, metrics):
    data, features, target = forecast_model.data, forecast_model.features, forecast_model.targets
    cv = cv_dict['cv']
    cv_split = list(cv.split(len(data), cv_dict['step'], cv_dict['gap']))
    
    results = {'mean_Y' : np.zeros((cv.train_size,1)),
               'mean_Y_pred' : np.zeros((cv.test_size, 1)), 
               'train_loss' : 0, 'test_loss' : 0, 'train_horizons' : [], 'test_horizons' : [],
               'mean_Y_train_pred' : np.zeros((cv.train_size, 1)), 'mean_Y_test' :np.zeros((cv.test_size, 1))}
    print(target)
    for train, test in cv_split:
        X_train, Y_train, X_test, Y_test = tsm.get_cv_matrices(data.loc[train], data.loc[test], features, target)
        fitted_model = model.fit(X_train, Y_train)
        Y_train_pred = np.reshape(fitted_model.predict(X_train), (len(X_train), 1))
        Y_pred = np.reshape(fitted_model.predict(X_test), (len(X_test), 1))
        results['train_loss'] += metrics(Y_train, Y_train_pred)
        results['test_loss'] += metrics(Y_test, Y_pred)
        results['mean_Y'] += Y_train
        results['mean_Y_train_pred'] += Y_train_pred
        results['mean_Y_test'] += Y_test
        results['mean_Y_pred'] += Y_pred
    results['train_loss'] #/= cv.n_splits
    results['test_loss'] #/= cv.n_splits
    results['mean_Y'] #/= cv.n_splits
    results['mean_Y_pred'] #/= cv.n_splits
    results['cv_dict'] = cv_dict
    return results

def run_cv(data, targets, horizons, CrossValid_params, ForecastModel_params, model, metrics):
    quality = {}
    features, date_time, prior_lag, post_lag = [ForecastModel_params['features'], ForecastModel_params['date_time'],
                                                ForecastModel_params['prior_lag'], ForecastModel_params['post_lag']]
    for tar in targets:
        quality[tar] = {'train_loss' : [], 'test_loss' : [], 'mean_Y' : [], 'mean_Y_pred' : [],
                        'mean_Y_train_pred' : [], 'mean_Y_test' : []}
    for hor in range(1, horizons + 1):
        for tar in targets:
            forecast_model = ForecastModel(date_time, data, features + targets, [tar], prior_lag = prior_lag, post_lag = post_lag)
            forecast_model.forecast_prep(ForecastModel_params['new_index'])
            train_size, test_size, min_period, step = [CrossValid_params['train_size'], CrossValid_params['test_size'],
                                                            CrossValid_params['min_period'], CrossValid_params['step']]
                                                            #CrossValid_params['gap']]
            cv_model = CrossValid(train_size = train_size, test_size = test_size, min_period = min_period)
            cv_dict = {'cv' : cv_model, 'step' : step, 'gap' : hor - 1}
            results = cross_valid(forecast_model, cv_dict, model, metrics) # (forecast_model, cv_dict, model, metrics)
            quality[tar]['train_loss'].append(results['train_loss'])
            quality[tar]['test_loss'].append(results['test_loss'])
            quality[tar]['mean_Y'].append(results['mean_Y'])
            quality[tar]['mean_Y_pred'].append(results['mean_Y_pred'])
            quality[tar]['mean_Y_train_pred'].append(results['mean_Y_train_pred'])
            quality[tar]['mean_Y_test'].append(results['mean_Y_test'])
            quality[tar]['cv_dict'] = cv_dict
            

def plot_cv_results(quality, horizons, results):
    for key in quality.keys():
        print(key)
        cv = results['cv_dict']['cv']
        train_index = list(range(cv.train_size))
        test_index = list(range(cv.train_size,cv.train_size + horizons))
        plt.plot(train_index,reduce(lambda a, b: a + b, quality[key]['mean_Y'])/horizons, label = 'mean_Y')
        plt.plot(train_index,reduce(lambda a, b: a + b, quality[key]['mean_Y_train_pred'])/horizons, alpha = .5, label = 'mean_Y_train_pred')
        plt.plot(test_index, np.reshape(quality[key]['mean_Y_test'], (horizons,1)),color = 'r', label = 'mean_Y_test')
        plt.plot(test_index, np.reshape(quality[key]['mean_Y_pred'],(horizons,1)), alpha = .5, label = 'mean_Y_pred')
        plt.legend()
        plt.show()