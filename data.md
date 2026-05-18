# 🎞 Data Preparation
## Getting the Dataset
* Download the raw video dataset you want. The supported options are:
    * [CC_WEB_VIDEO](http://vireo.cs.cityu.edu.hk/webvideo/)
    * [VCDB](https://fvl.fudan.edu.cn/dataset/vcdb/list.htm)
    * [FIVR](http://ndd.iti.gr/fivr/)  
    * [EVVE](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6619162)  


* You should contact the author about the missing video that occurs during the download process.
* The raw video data should be located like the structure below. 

* **But preparing raw video is not essential. We provide the features, we used.**

~~~~
├── videos
   ├── fivr
      └── videos
         ├── video_1
         ├── video_2
         └── ...
   ├── cc_web
      └── videos
         ├── video_1
         ├── video_2
         └── ...
   ├── evve
      └── videos
         ├── video_1
         ├── video_2
         └── ...
~~~~
  

