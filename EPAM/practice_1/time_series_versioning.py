import pickle
from datetime import datetime
from pandas import DataFrame as pandas_DataFrame
from pandas import read_csv as pandas_read_csv

class nested_data:
    def __init__(self, data, features, targets, date_time, model = None, accuracy = None):
        self.data = data.copy()
        self.features = features.copy()
        self.targets = targets.copy()
        self.date_time = date_time
        self.model, self.accuracy = model, accuracy
    def get_data(self):
        return self.data.copy()
    def get_features(self):
        return self.features.copy()
    def get_targets(self):
        return self.targets.copy()
    def get_time(self):
        return self.date_time.copy()
    def get_accuracy(self):
        return self.accuracy
    def info(self):
        print('FEATURES:\n' + self.features)
        print('TARGETS:\n' + self.targets)
        print('DATE_TIME: ' + self.date_time)

class DataVersions:
    def __init__(self):
        self.versions = {}
        self.csvs = {}
    
    def get(self, key):
        return self.versions[key]        

    def push(self, data, features, targets, model = None, accuracy = None, key = False):
        current_time = datetime.now().strftime('%d_%m_%Y_%H:%M')
        self.versions[key if key else current_time] = nested_data(data.copy(), features.copy(), targets.copy(), current_time)

    def save_as_csv(self, key, name = False, delete = False):
        dataframe = pandas_DataFrame(data = {'data' : self.data, 'accuracy' : self.accuracy, 'model' : self.model})
        dataframe.to_csv(name if name else key + '.csv', index = False)
        # self.versions[key].data.to_csv(name if name else key + '.csv', index = False)
        # self.csvs[name if name else key + '.csv'] = True
        # if delete:
        #     del self.versions[key]
        #     self.csvs[name if name else key + '.csv'] = False

    def save_as_dataframe(self, data, accuracy, model, key, name = False, delete = False):
        dataframe = pandas_DataFrame(data = {'data' : data, 'accuracy' : accuracy, 'model' : model})
        dataframe.to_csv(name if name else key + '.csv', index = False)

    def get_from_csv(self, name):
        try:
            data = pandas_read_csv(name)
        except FileNotFoundError:
            print(r"File '" + name +r"' not found")
        else:
            return data

    def delete(self, key):
        del self.versions[key]