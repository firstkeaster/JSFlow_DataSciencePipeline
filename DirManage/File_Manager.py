import os
from datetime import datetime

nowpath = os.path.abspath('.')

class File_Manager(object):
    def init(self, file_path):
        self.nowpath = os.path.abspath('.')
        if not os.path.exists(file_path):
            try:
                os.makedirs(file_path)
                print('Making directory successful!')
            except:
                print('Failure making working directory!')
                return
        if not os.path.exists(file_path+'/documents'):
            os.makedirs(file_path+'/documents')
        if not os.path.exists(file_path+'/dataSources'):
            os.makedirs(file_path+'/dataSources')
        if not os.path.exists(file_path+'/dataSets'):
            os.makedirs(file_path+'/dataSets')
        if not os.path.exists(file_path+'/sourceCode'):
            os.makedirs(file_path+'/sourceCode')
        if not os.path.exists(file_path+'/log'):
            os.makedirs(file_path+'/log')
            #Write in append mode
        if not os.path.exists(file_path+'/log/general.log'):
            log_str = 'Working directory created!'
        else:
            log_str = 'Working directory repaired!'
        log_writer = open(file_path+'/log/general.log','a')
        log_writer.write(str(datetime.now()))
        log_writer.write(log_str)
        log_writer.write('\n')
        log_writer.close()
    
    def test(self, file_path):
        issuer = False
        if not os.path.exists(file_path):
            print('No such directory!')
            issuer = True
            return
        if not os.path.exists(file_path+'/documents'):
            print('"/documents" lost!')
            issuer = True
        if not os.path.exists(file_path+'/dataSources'):
            print('"/dataSources" lost!')
            issuer = True
        if not os.path.exists(file_path+'/dataSets'):
            print('"/dataSets" lost!')
            issuer = True
        if not os.path.exists(file_path+'/sourceCode'):
            print('"/sourceCode" lost!')
            issuer = True
        if not os.path.exists(file_path+'/log'):
            print('"/log" lost!')
            issuer = True
        if issuer == False:
            print('All systems normal! Keep walking!')
    
    def data_adder(self, file_path, data_path, file_type=None):
        '''
        file_type: 'train', 'val', 'test', 'label'
        Planning: manage several models, each have a independent
        list file.
        '''
        switcher = {
            'train': file_path + '/documents/train_datalist.txt',
            'val': file_path + '/documents/val_datalist.txt',
            'test': file_path + '/documents/test_datalist.txt',
            'label': file_path + '/documents/label_datalist.txt'
        }
        f = open(switcher[file_type],'a')
        f.write(data_path)
        f.write('\n')
        f.close()

    def data_remover(self, file_path, data_path, file_type=None):
        switcher = {
            'train': file_path + '/documents/train_datalist.txt',
            'val': file_path + '/documents/val_datalist.txt',
            'test': file_path + '/documents/test_datalist.txt',
            'label': file_path + '/documents/label_datalist.txt'
        }
        f = open(switcher[file_type],'r+')
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if data_path not in line:
                f.write(line)
        f.truncate()
        f.close()

    


        

