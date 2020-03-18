
# introdution
We design a face recognition for political person, which include 274 classes. we have a api code, you can test images from internet.

# prepare dataset
+ you should make a data struct
>data/class1/sdsd.jpg
>
>data/class1/sdsds.jpg
>
>...
>
>data/class2/asas.jpg
+ revise `make_datasettxt.py` the data_dir...
+ run `python make_datasettxt.py` to generate train_dataset.txt
like this
```
./politicalfacetest1align/ailigengyimingbahai/8c52e334b279cab5.jpg 0
./politicalfacetest1align/ailigengyimingbahai/94cf87f0a3a4898f.jpg 0
./politicalfacetest1align/ailigengyimingbahai/9bb44deba44be085.jpg 0
./politicalfacetest1align/ailigengyimingbahai/a6998e30c79f318d.jpg 0
./politicalfacetest1align/aimaniuaiermakelong/89a594a3f5ed6314.jpg 1
./politicalfacetest1align/aimaniuaiermakelong/8ef826f36027f30c.jpg 1
./politicalfacetest1align/aimaniuaiermakelong/93c952bb4a4287b7.jpg 1
./politicalfacetest1align/aimaniuaiermakelong/94f5496a56db2b81.jpg 1
./politicalfacetest1align/baienpei/89a7e227da4aa791.jpg 2
./politicalfacetest1align/baienpei/932e3cc590373a6e.jpg 2
./politicalfacetest1align/baienpei/9836656366cc93cd.jpg 2
./politicalfacetest1align/baienpei/9b5962a6506a3fe1.jpg 2
...
```
# train
+ run `python train.py`
# pretrain model
+ the political person face detection model is []()
# test
+ The face_test_client.py is a pyqt code,you can `python face_test_client`, and enter the port 91, enter a images url to test. The client shows:

![](https://gongkai-1257632417.cos.ap-chengdu.myqcloud.com/client.png) 

+ the return value is a json.
```json
{"object": [
    {"grade": 0.5884853601455688, 
    "class": "xijinping", 
    "pos": {
      "y": 138, 
      "x": 474, 
      "w": 89, 
      "h": 89
    }},
    {"grade": 0.5833901166915894, 
    "class": "hujintao", 
    "pos": {
      "y": 167, 
      "x": 246, 
      "w": 85, 
      "h": 85}}
    ],
  "img_url": "http://news.cnr.cn/dj/20160914/W020160914388600058399.jpg"
}
```
+ If you want to pull on your own server,you should write a server code, we give a demo `face_test_server.py`

# References
https://github.com/deepinsight/insightface
https://github.com/auroua/InsightFace_TF
