#encoding=utf-8
import os

data_dir = './politicalfacetrain1align/'

f = open('./train_dataset.txt','w')
f2 = open('./data_label.txt','w')
count = 0
for index_class in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir,index_class)):
        img_full_path = '%s/%s'%(os.path.join(data_dir, index_class),img_path)
        f.write("%s %i\n"%(img_full_path,count))
    f2.write("%s %i\n"%(index_class,count))
    print("%s %i"%(index_class,count))
    count += 1
print('ok')
f.close()
data_dir = "./politicalfacetest1align/"
f = open('./val_dataset.txt','w')
count = 0
for index_class in os.listdir(data_dir):
    if index_class == 'N_sample':
        continue
    for img_path in os.listdir(os.path.join(data_dir,index_class)):
        img_full_path = '%s/%s'%(os.path.join(data_dir, index_class),img_path)
        f.write("%s %i\n"%(img_full_path,count))
    #f2.writelines('%s %i'%(index_class,count))
    print('%s %i'%(index_class,count))
    count += 1
print('ok')
f.close()

