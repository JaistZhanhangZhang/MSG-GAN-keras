import numpy as np
import os
from PIL import Image
rootdir="../images1024x1024/"
groups=os.listdir(rootdir)
file_list=[]
for group in groups:
    file_root=rootdir+group+"/"
    flist=os.listdir(file_root)
    for file in flist:
        filename=file_root+file
        if filename[-4:]==".png":
            file_list.append(filename)
#print(len(file_list))

def dataloader(imgs=file_list,batch_size=4):
    im_1024_list=[]
    im_512_list=[]
    im_256_list=[]
    im_128_list=[]
    im_64_list=[]
    im_32_list=[]
    im_16_list=[]
    im_8_list=[]
    im_4_list=[]

    while True:
        for i in np.random.permutation(len(imgs)):
            im_1024 = Image.open(imgs[i])
            im_512_list.append((np.asarray(im_1024.resize((512,512))) - 127.5 ) / 127.5)
            im_256_list.append( (np.asarray(im_1024.resize((256,256))) - 127.5 ) / 127.5)
            im_128_list.append( (np.asarray(im_1024.resize((128,128))) - 127.5 ) / 127.5)
            im_64_list.append( (np.asarray(im_1024.resize((64,64)))   - 127.5 ) / 127.5)
            im_32_list.append( (np.asarray(im_1024.resize((32,32)))   - 127.5 ) / 127.5)
            im_16_list.append( (np.asarray(im_1024.resize((16,16)))   - 127.5 ) / 127.5)
            im_8_list.append( (np.asarray(im_1024.resize((8,8)))     - 127.5 ) / 127.5)
            im_4_list.append( (np.asarray(im_1024.resize((4,4)))     - 127.5 ) / 127.5)
            im_1024_list.append( (np.asarray(im_1024) - 127.5 ) / 127.5)


            #batch_imgs.append([im_4,im_8,im_16,im_32,im_64,im_128,im_256,im_512,im_1024])
            if len(im_4_list) == batch_size:
                im_1024_list=np.asarray(im_1024_list)
                im_512_list=np.asarray(im_512_list)
                im_256_list=np.asarray(im_256_list)
                im_128_list=np.asarray(im_128_list)
                im_64_list=np.asarray(im_64_list)
                im_32_list=np.asarray(im_32_list)
                im_16_list=np.asarray(im_16_list)
                im_8_list=np.asarray(im_8_list)
                im_4_list=np.asarray(im_4_list)

                y_batch=[im_4_list,im_8_list,im_16_list,im_32_list,im_64_list,im_128_list,im_256_list,im_512_list,im_1024_list]
                yield y_batch
                im_1024_list=[]
                im_512_list=[]
                im_256_list=[]
                im_128_list=[]
                im_64_list=[]
                im_32_list=[]
                im_16_list=[]
                im_8_list=[]
                im_4_list=[]