//
// Created by Himi on 2022/8/3.
//

#import <opencv2/opencv.hpp>
#import <TensorFlowLiteC/TensorFlowLiteC.h>

#include <stdint.h>
#include <chrono>

using namespace cv;
using namespace std;
#define DART_API extern "C" __attribute__((visibility("default"))) __attribute__((used))

DART_API int32_t testByFfi(int32_t x, int32_t y) {
    
    //opencv test
    Mat testMat(Mat::zeros(2,2,CV_8U));
    testMat += 5;
    int result =  testMat.at<uint8_t>(0,1) * testMat.at<uint8_t>(1,1);
    
    
    //tflite test
    cout<< TfLiteVersion() << endl;
    
    return result;
}


long getTimestamp(){
    auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch());
    return ms.count();
}


template <class T>
void clearVector(vector<T>& vt)
{
    vector<T> vtTemp;
    vtTemp.swap(vt);
}


class ChohoKalmanStablizer
{
    private:
        KalmanFilter KF;
        uint64_t vacantCount;
        float old_x, old_y;
        void initKalmanFilter(const Point_<float> *init_pos){
            KF.init(4, 4, 0, CV_32F);
            KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,0.3,0,   0,1,0,0.3,  0,0,1,0,  0,0,0,1);
            KF.statePre.at<float>(0) = init_pos->x;
            KF.statePre.at<float>(1) = init_pos->y;
            KF.statePre.at<float>(2) = 0;
            KF.statePre.at<float>(3) = 0;
            setIdentity(KF.measurementMatrix);
            setIdentity(KF.processNoiseCov, Scalar(0.01f, 0.01f, 0.0001f, 0.0001f));
            setIdentity(KF.measurementNoiseCov, Scalar(0.3f, 0.3f, 1e-1, 1e-1));
            setIdentity(KF.errorCovPost, Scalar(0.01f, 0.01f, 0.1f, 0.1f));
            KF.statePost.at<float>(0) = init_pos->x;
            KF.statePost.at<float>(1) = init_pos->y;
            KF.statePost.at<float>(2) = 0;
            KF.statePost.at<float>(3) = 0;
        }
    public:
        ChohoKalmanStablizer(): vacantCount(1000), old_x(-99999.0f), old_y(-99999.0f){}
        ChohoKalmanStablizer(const Point_<float> &init_pos): vacantCount(0), old_x(init_pos.x), old_y(init_pos.y){
            initKalmanFilter(&init_pos);
        }
        void resetKalman(){
            vacantCount = 1000;
            old_x = -99999.0f;
            old_y = -99999.0f;
        }
        ~ChohoKalmanStablizer(){}
        const Point_<float> getEstPt(const Point_<float> *pt){
            if(pt == nullptr){
                vacantCount += 1;
                if (vacantCount > 2){
                    old_x = -99999.0f;
                    old_y = -99999.0f;
                    return Point_<float>(-99999.0, -99999.0);
                }
                return Point_<float>(KF.statePost.at<float>(0), KF.statePost.at<float>(1));
            }
            if(vacantCount > 2){
                initKalmanFilter(pt);
            } else {
                Mat_<float> measurement(4, 1);
                measurement(0) = pt->x;
                measurement(1) = pt->y;
                measurement(2) = ((old_x > -1.0)?pt->x - old_x:0.0) / (1.0 + vacantCount);
                measurement(3) = ((old_y > -1.0)?pt->y - old_y:0.0) / (1.0 + vacantCount);
                KF.predict();
                KF.correct(measurement);
            }
            old_x = pt->x;
            old_y = pt->y;
            vacantCount = 0;
            return Point_<float>(KF.statePost.at<float>(0), KF.statePost.at<float>(1));
        }
};
static ChohoKalmanStablizer kalman;

static vector<const char*>labels= {
    "upper", "lower", "lr", "front", "overjet", "bite",
    "u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8",
    "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"
};


typedef struct {
    //pre
    int offsets_l;
    int offsets_t;
    int offsets_r;
    int offsets_b;
    int rect_x;
    int rect_y;
    int rect_w;
    int rect_h;
    int ori_size_w;
    int ori_size_h;
    float scale;
    uint8_t * preImgBytes_uint8;
    float * preImgBytes_float;
    int preImgBytesLen;
    int isInt8TFLite;
    int design_width;
    int design_height;
    //last
    uint8_t* imgDataList;
    const char* imgPath;
    int rows;
    int cols;
    float outputInt8TFLiteScale;
    int outputInt8TFLiteZeroPoint;
} Struct2PreTFLite;




int clamp(int lower, int higher, int val){
    if(val < lower)
        return 0;
    else if(val > higher)
        return 255;
    else
        return val;
}

int getRotatedImageByteIndex(int x, int y, int rotatedImageWidth){
    return rotatedImageWidth*(y+1)-(x+1);
}

Mat convertYuvDatasToImage(uint8_t *plane0, int width, int height, uint8_t *plane1, uint8_t *plane2, int bytesPerRow, int bytesPerPixel, int yRowStride, uint8_t **imageData){
    int w, h, uvIndex, index,yIndex;

    *(imageData) =  new uint8_t[width*height*3];

    for(w = 0; w < width; w++){
        for(h = 0; h < height; h++){
            
            uvIndex = bytesPerPixel * (w/2) + bytesPerRow * (h/2);
            index = h*width+w;
            yIndex = h*yRowStride+w;

            (*imageData)[index * 3] = plane0[yIndex];
            (*imageData)[index * 3 + 1] = plane1[uvIndex];
            (*imageData)[index * 3 + 2] = plane2[uvIndex];
        }
    }
    Mat img = Mat(height, width, CV_8UC3, *imageData);
    cvtColor(img, img, COLOR_YUV2RGB);
    return img;
}


//imgDatasType:
//0 = rgb 格式
//1 = rbga8888 格式
//2 = yuv420 格式
Struct2PreTFLite preTFLiteProcess(uint8_t* imgDatas,int width ,int height,const char* imgPathStr,int isInt8TFLite,float inputInt8TFLiteScale,
                      int inputInt8TFLiteZeroPoint,int design_width,int design_height,int isCheckTwoPoint,int imgDatasType,uint8_t* plane1,uint8_t* plane2, int bytesPerRow, int bytesPerPixel, int yRowStride, bool isUseNewModal){
    
    Struct2PreTFLite callbackObj;
    callbackObj.imgDataList = imgDatas;
    
    Mat oldImgMat;
    Mat oldImgMatTemp;
    Size oldImgSize;
    if(strcmp(imgPathStr, "")==0){
        if(imgDatasType == 1){
            //imgDatas-> bgra8888 -> rgb
            oldImgMat = Mat(Size(width,height),CV_8UC4,imgDatas);
            cvtColor(oldImgMat, oldImgMat, COLOR_BGRA2RGB);
            callbackObj.imgDataList = new uint8_t[height*width * 3];
            for (int i = 0; i < height * width * 3; i++)
            {
                callbackObj.imgDataList[i] = (uint8_t)oldImgMat.at<Vec3b>(i / (width * 3), (i % (width * 3)) / 3)[i % 3];
            }
            delete [] imgDatas;
        }else if(imgDatasType == 2){
            //imgDatas-> yuv420 -> rgb
            oldImgMat =  convertYuvDatasToImage(imgDatas, width, height,plane1,plane2,bytesPerRow,bytesPerPixel,yRowStride,&(callbackObj.imgDataList));
            delete [] imgDatas;
        }else{
            //imgDatas -> rgb
            oldImgMat = Mat(Size(width,height),CV_8UC3,imgDatas);
        }
        oldImgSize = Size(width,height);
    }else{
        //imread 默认 BGR 三通道
        oldImgMat = imread(imgPathStr);
        oldImgSize = Size(oldImgMat.cols,oldImgMat.rows);
        
        if(isUseNewModal){
            cvtColor(oldImgMat, oldImgMat, COLOR_BGR2RGB);
        }
        if(isCheckTwoPoint){
            cvtColor(oldImgMat, oldImgMat, COLOR_BGR2RGB);
        }
    }
    
    Size netImgSize =Size(design_width,design_height);
    Size newImgSize;
    float scaleRatio = 0;
    
    if(float(netImgSize.width)/float(oldImgSize.width) < float(netImgSize.height)/float(oldImgSize.height)){
        scaleRatio =float(netImgSize.width)/float(oldImgSize.width);
        newImgSize = Size(netImgSize.width,int(oldImgSize.height*scaleRatio));
    }else{
        scaleRatio =float(netImgSize.height)/float(oldImgSize.height);
        newImgSize = Size(int(oldImgSize.width*scaleRatio),netImgSize.height);
    }
    Mat newImgMat;
    resize(oldImgMat, newImgMat, Size(newImgSize.width,newImgSize.height));
    
    //
    int delta_w = netImgSize.width - newImgSize.width;
    int delta_h = netImgSize.height - newImgSize.height;
    int top = delta_h/2;
    int left = delta_w/2;
    int bottom = delta_h-(delta_h/2);
    int right = delta_w-(delta_w/2);
    
    //补充空白像素
    Mat copyMakeMat;
    copyMakeBorder(newImgMat, copyMakeMat, top, bottom, left, right, BORDER_CONSTANT);
    
    //像素归一化
    Mat reslutImgMat;
    copyMakeMat.convertTo(reslutImgMat, CV_32F, 1.0 / 255, 0);
    
    //int8的tflite
    if(isInt8TFLite){
        reslutImgMat /= inputInt8TFLiteScale;
        reslutImgMat += inputInt8TFLiteZeroPoint;
        reslutImgMat.convertTo(reslutImgMat, CV_8U);
    }
    
    callbackObj.offsets_l = left;
    callbackObj.offsets_t = top;
    callbackObj.offsets_r = newImgSize.width+left;
    callbackObj.offsets_b = newImgSize.height+top;
    callbackObj.rect_x = 0;
    callbackObj.rect_y = 0;
    callbackObj.rect_w = oldImgSize.width;
    callbackObj.rect_h = oldImgSize.height;
    callbackObj.ori_size_w = oldImgSize.width;
    callbackObj.ori_size_h = oldImgSize.height;
    callbackObj.scale = scaleRatio;

    int matTotal = reslutImgMat.rows*reslutImgMat.cols*3;
    
    if(isInt8TFLite){
        uint8_t * uDatas = new uint8_t[matTotal];
        memcpy(uDatas, reslutImgMat.data, matTotal*sizeof(uint8_t));
        callbackObj.preImgBytes_uint8 = uDatas;//(uint8)
        callbackObj.preImgBytes_float = nullptr;//(float32)
    }else{
        float * uDatas = new float[matTotal];
        memcpy(uDatas, reslutImgMat.data, matTotal*sizeof(float));
        callbackObj.preImgBytes_float = uDatas;//(float32)
        callbackObj.preImgBytes_uint8 = nullptr;//(uint8)
    }
    
    callbackObj.preImgBytesLen = matTotal;
    callbackObj.isInt8TFLite = isInt8TFLite;
    callbackObj.design_width = design_width;
    callbackObj.design_height = design_height;

    return callbackObj;
}



Mat createLastMatProcess(Struct2PreTFLite preObjDic,void * tflite_datas,int tflite_rows2,int tflite_cols2){
    int tflite_rows = preObjDic.rows;
    int tflite_cols = preObjDic.cols;
    if(tflite_rows2 && tflite_cols2){
        tflite_rows =tflite_rows2;
        tflite_cols = tflite_cols2;
    }
    //output
    Mat grid;
    //int8的tflite
    if(preObjDic.isInt8TFLite){
        grid = Mat(Size(tflite_rows,tflite_cols), CV_8U,tflite_datas);
        
        float outputInt8TFLiteScale = preObjDic.outputInt8TFLiteScale;
        int int8TFLiteOutputZeroPoint = preObjDic.outputInt8TFLiteZeroPoint;
        grid.convertTo(grid, CV_32F);
        grid -= int8TFLiteOutputZeroPoint;
        grid *= outputInt8TFLiteScale;
        
    }else{
        grid = Mat(Size(tflite_rows,tflite_cols), CV_32F,tflite_datas);
    }

    return grid;
}


Mat backOriImagePosition(Struct2PreTFLite preObjDic,Mat xyxy){
    int design_width = preObjDic.design_width;
    int design_height = preObjDic.design_height;
    //预处理图片的预处理数据,
    int offsets[4] = {preObjDic.offsets_l,preObjDic.offsets_t,preObjDic.offsets_r,preObjDic.offsets_b};
    int rect[4] = {preObjDic.rect_x,preObjDic.rect_y,preObjDic.rect_w,preObjDic.rect_h};
    float scale = preObjDic.scale;
    //tflite 都是0~1的值，先还原坐标点
    xyxy.col(0) *= design_width;
    xyxy.col(1) *= design_height;
    if(xyxy.cols>2){
        xyxy.col(2) *= design_width;
        xyxy.col(3) *= design_height;
    }
    //image back
    xyxy.col(0) -= offsets[0];
    xyxy.col(1) -= offsets[1];
    if(xyxy.cols>2){
        xyxy.col(2) -= offsets[0];
        xyxy.col(3) -= offsets[1];
    }
    xyxy /= scale;
    xyxy.col(0) += rect[0];
    xyxy.col(1) += rect[1];
    if(xyxy.cols>2){
        xyxy.col(2) += rect[0];
        xyxy.col(3) += rect[1];
    }
    return xyxy;
}
typedef struct {
    int isBlur;
    int checkLapMax;
    int checkLapMin;
    float imgLaplacianCount;
    float checkCannyCount;
    float imgCannyCount;
}Struct2LastTFLiteBlurInfo;


typedef struct {
    float* rects;
    int rectEleCount;
    int rectsLength;
    int num_classes;
    Struct2LastTFLiteBlurInfo blurInfo;
    int importNetOriImgW;
    int importNetOriImgH;
}Struct2LastTFLite;


Struct2LastTFLiteBlurInfo checkIsBlur(Struct2PreTFLite preObjDic,Point lt, Point rb){
    
    int ori_size[2] = {preObjDic.ori_size_w,preObjDic.ori_size_h};
    //检查是否模糊(没有resize过的原图,且是灰色单通道)
    Mat oldImgMat;
    if(strcmp(preObjDic.imgPath, "")==0){
        oldImgMat = Mat(Size(ori_size[0],ori_size[1]),CV_8UC3,preObjDic.imgDataList);
        cvtColor(oldImgMat, oldImgMat, COLOR_BGR2GRAY);//灰色单通道
    }else{
        oldImgMat = imread(preObjDic.imgPath,IMREAD_GRAYSCALE);//灰色单通道
    }
    
    double laplacian_check = -1;
    double canny_check = -1;
    bool imgIsBlur = true;
    int checkLapMax = 235;
    int checkLapMin = 80;
    double checkCannyCount = 3.9;
    if(lt.x !=0 && lt.y !=0 && rb.x !=0 && rb.y!=0){
        Mat tempLapMat;
        Laplacian(oldImgMat, tempLapMat, CV_16S);
        Mat tempScaMat;
        convertScaleAbs(tempLapMat, tempScaMat);
        minMaxIdx(tempScaMat,0,&laplacian_check);
        if(laplacian_check>checkLapMax){
            imgIsBlur = false;
        }else if(laplacian_check<checkLapMin){
            imgIsBlur = true;
        }else{
            if(lt.x<=0){lt.x = 0;}
            if(lt.y<=0){lt.y = 0;}
            if(rb.x<=0){rb.x = 0;}
            if(rb.y<=0){rb.y = 0;}
            if(lt.x>=oldImgMat.cols){lt.x = oldImgMat.cols;}
            if(rb.x>=oldImgMat.cols){rb.x = oldImgMat.cols;}
            if(lt.y>=oldImgMat.rows){lt.y = oldImgMat.rows;}
            if(rb.y>=oldImgMat.rows){rb.y = oldImgMat.rows;}
            
            Mat rectMaxMat = oldImgMat(Range(lt.y,rb.y),Range(lt.x,rb.x));
            Mat edges;
            Canny(rectMaxMat, edges, 100, 200);

            canny_check = countNonZero(edges)*1000.0/(edges.rows*edges.cols);
            if(canny_check>checkCannyCount){
                imgIsBlur = false;
            }else{
                imgIsBlur = true;
            }
        }
    }
    
    Struct2LastTFLiteBlurInfo blurInfo;
    blurInfo.isBlur = imgIsBlur;
    blurInfo.checkLapMax  = checkLapMax;
    blurInfo.checkLapMin = checkLapMin;
    blurInfo.imgLaplacianCount = laplacian_check;
    blurInfo.checkCannyCount = checkCannyCount;
    blurInfo.imgCannyCount = canny_check;
    
    return blurInfo;
}




static vector<const char*> newLabels= {
//  "upper", "lower", "ul","kps","moli",
  "0", "1", "2","3","4",
  "11", "12", "13","14", "15", "16","17", "18",
  "21", "22", "23","24", "25", "26","27", "28",
  "31", "32", "33","34", "35", "36","37", "38",
  "41", "42", "43","44", "45", "46","47", "48",
};

Struct2LastTFLite newLastTFLiteProcess(void* tflite_datas,Struct2PreTFLite preObjDic){
  Mat grid = createLastMatProcess(preObjDic, tflite_datas, 0, 0);

  //获取每个框体对应22个牙齿的得分 = 置信度*22个牙齿
  //参与点乘的两个Mat矩阵的数据类型只能是CV_32F、CV_64FC1、CV_32FC2、CV_64FC2这4种类型中的一种。
  //保证维度相同再相乘
  int class_num = (int)newLabels.size();
  int reg_num = 6;
  int conf_index = 6;

  // 角度信息，这里是 cos sin
  Mat direction = grid(Range::all(),Range(4, 6));

  // 该框是否包含物体的置信度
  Mat conf = repeat(grid.col(conf_index), 1, class_num);

  // 各个类别的置信度
  int cls_index = conf_index + 1;
  Mat cls = grid(Range::all(),Range(cls_index, cls_index + class_num));

  // 各个框的回归值 参数
  int reg_index = cls_index + class_num;
  Mat reg = grid(Range::all(),Range(reg_index, reg_index + reg_num));

  // conf × cls 为最终的置信度
  Mat probs = conf.mul(cls);

  //获取每个框对应最高分类别的下标
  Mat labelIndexs = Mat(grid.rows, 1, CV_32F);
  //获取每个框对应最高分牙齿的下标的分值
  Mat scores = Mat(grid.rows, 1, CV_32F);
  double max = 0;
  int maxIndex[2]={0,0};
  for (int i = 0; i<probs.rows; i++) {
    minMaxIdx(probs.row(i),0,&max,0,maxIndex);
    labelIndexs.at<float>(i,0) = maxIndex[1];
    scores.at<float>(i,0) = max;
  }

  //xywh，由于int8的问题，这里的数值已经被归一化
  Mat yolo_xywh = grid(Range::all(),Range(0,4));
  //image back
  //变化坐标点用于后续画Rect统一左上角和右下角
  Mat xyxy = Mat(yolo_xywh.rows, yolo_xywh.cols, CV_32F);
  xyxy.col(0) = yolo_xywh.col(0) - yolo_xywh.col(2)/2;
  xyxy.col(1) = yolo_xywh.col(1) - yolo_xywh.col(3)/2;
  xyxy.col(2) = yolo_xywh.col(0) + yolo_xywh.col(2)/2;
  xyxy.col(3) = yolo_xywh.col(1) + yolo_xywh.col(3)/2;

  backOriImagePosition(preObjDic,xyxy);

  Mat xywh = xyxy.clone();
  xywh.col(2) = xywh.col(2) - xywh.col(0);
  xywh.col(3) = xywh.col(3) - xywh.col(1);

  //预测
  int bbox_num = xywh.rows;
  vector<Rect2d> boxes(bbox_num);
  for(int i = 0;i<xywh.rows;i++){
    boxes[i] = (Rect2d(xywh.at<float>(i,0),xywh.at<float>(i,1),xywh.at<float>(i,2),xywh.at<float>(i,3)));
  }

  vector<int> indices;
  vector<int> labels;
  bool class_wise = 1;
  if (class_wise){
      vector<int> class_indices;
      for(int i=0; i < class_num; i++){
          Mat tempScore = probs(Range::all(),Range(i,i+1));
        dnn::NMSBoxes(boxes, tempScore, 0.2, 0.4, class_indices);
        for(auto v: class_indices){
          indices.push_back(v);
          labels.push_back(i);
        }
        class_indices.clear();
      }
  }else{
     vector<int> class_indices;
     dnn::NMSBoxes(boxes, scores, 0.2, 0.4, class_indices);
     for (auto v: class_indices){
        indices.push_back(v);
        labels.push_back((int)labelIndexs.at<float>(v, 0));
      }
  }
  //

  clearVector(boxes);

  //整理识别出来的目标矩形等信息
  // x, y, w, h, theta, prob, label, [..., reg_num]
  int num_features = 6 + 1 + reg_num;
  float *rectsArray = new float[indices.size()*num_features];

  //取出最大的框体左上右下
  Point lt,rb;
    
  for (int i=0; i<indices.size(); i++) {
    int _index = labels[i];

    int _k = indices[i];
    //第几颗牙
    //评分(可信度)
    float _score = scores.at<float>(_k,0);
    //牙齿矩形框位置
    float r_l = xyxy.at<float>(_k,0);
    float r_t = xyxy.at<float>(_k,1);
    float r_r = xyxy.at<float>(_k,2);
    float r_b = xyxy.at<float>(_k,3);
    float _theta = 0; // TODO
      
    //拿到最大的框体
    if(r_r-r_l>rb.x-lt.x){
        lt = Point(r_l,r_t);
        rb = Point(r_r,r_b);
    }

    rectsArray[i*num_features] = r_l;
    rectsArray[i*num_features+1] = r_t;
    rectsArray[i*num_features+2] = r_r;
    rectsArray[i*num_features+3] = r_b;
    rectsArray[i*num_features+4] = _theta;
    rectsArray[i*num_features+5] = _score;
    rectsArray[i*num_features+6] = atoi(newLabels[_index]);

    for (int j = 0; j < reg_num; j++){
      rectsArray[i * num_features + 7 + j] = reg.at<float>(_k, j);
    }
  }
  Struct2LastTFLite lastObj;
  lastObj.rects = rectsArray;
  lastObj.rectEleCount =num_features;
  lastObj.rectsLength = (int)indices.size()* num_features;
  lastObj.blurInfo = checkIsBlur(preObjDic, lt, rb);
  lastObj.importNetOriImgW = 0;
  lastObj.importNetOriImgH = 0;

  clearVector(indices);
  return lastObj;
}




Struct2LastTFLite lastTFLiteProcess(void* tflite_datas,Struct2PreTFLite preObjDic){
    
    Mat grid = createLastMatProcess(preObjDic, tflite_datas, 0, 0);
    
    //获取每个框体对应22个牙齿的得分 = 置信度*22个牙齿
    //参与点乘的两个Mat矩阵的数据类型只能是CV_32F、CV_64FC1、CV_32FC2、CV_64FC2这4种类型中的一种。
    //保证维度相同再相乘
    Mat newScore = repeat(grid.col(4), 1, grid.cols-5);
    Mat tooths = grid(Range::all(),Range(5,grid.cols));
    Mat probs = newScore.mul(tooths);
    //获取每个框对应最高分牙齿的下标
    Mat labelIndexs = Mat(grid.rows, 1, CV_32F);
    //获取每个框对应最高分牙齿的下标的分值
    Mat scores = Mat(grid.rows, 1, CV_32F);
    double max = 0;
    int maxIndex[2]={0,0};
    for (int i = 0; i<probs.rows; i++) {
        minMaxIdx(probs.row(i),0,&max,0,maxIndex);
        labelIndexs.at<float>(0,i) = maxIndex[1];
        scores.at<float>(0,i) = max;
    }
    
    //xywh
    Mat yolo_xywh = grid(Range::all(),Range(0,4));
    //image back
    //变化坐标点用于后续画Rect统一左上角和右下角
    Mat xyxy = Mat(yolo_xywh.rows, yolo_xywh.cols, CV_32F);
    xyxy.col(0) = yolo_xywh.col(0) - yolo_xywh.col(2)/2;
    xyxy.col(1) = yolo_xywh.col(1) - yolo_xywh.col(3)/2;
    xyxy.col(2) = yolo_xywh.col(0) + yolo_xywh.col(2)/2;
    xyxy.col(3) = yolo_xywh.col(1) + yolo_xywh.col(3)/2;
    
    backOriImagePosition(preObjDic,xyxy);
    
    Mat xywh = xyxy.clone();
    xywh.col(2) = xywh.col(2) - xywh.col(0);
    xywh.col(3) = xywh.col(3) - xywh.col(1);

    //预测
    vector<Rect2d> boxes;
    for(int i = 0;i<xywh.rows;i++){
        boxes.push_back(Rect2d(xywh.at<float>(i,0),xywh.at<float>(i,1),xywh.at<float>(i,2),xywh.at<float>(i,3)));
    }
    vector<int> indices;
    //第一个参数：预测框的尺寸（预测框左上角和右下角的尺寸）
    //第二个参数：预测中的的置信度得分
    dnn::NMSBoxes(boxes, scores, 0.25, 0.35, indices);
    
    clearVector(boxes);
    
    //取出最大的框体左上右下
    Point lt,rb;
    
    //分类长度
    int num_classes = probs.cols;
    
    //整理识别出来的目标矩形等信息
    float *rectsArray = new float[indices.size()*6];
    
    int num_features = 6;
    for (int i=0; i<indices.size(); i++) {
        int _k = indices[i];
        //第几颗牙
        int _index = (int)labelIndexs.at<float>(_k,0);
        //评分(可信度)
        float _score = scores.at<float>(_k,0);
        //牙齿矩形框位置
        float r_l = xyxy.at<float>(_k,0);
        float r_t = xyxy.at<float>(_k,1);
        float r_r = xyxy.at<float>(_k,2);
        float r_b = xyxy.at<float>(_k,3);
        
        //拿到最大的框体
        if(r_r-r_l>rb.x-lt.x){
            lt = Point(r_l,r_t);
            rb = Point(r_r,r_b);
        }
        
        rectsArray[i*num_features] = r_l;
        rectsArray[i*num_features+1] = r_t;
        rectsArray[i*num_features+2] = r_r;
        rectsArray[i*num_features+3] = r_b;
        rectsArray[i*num_features+4] = _score;
        rectsArray[i*num_features+5] = _index;
    }
    Struct2LastTFLite lastObj;
    lastObj.rects = rectsArray;
    lastObj.rectEleCount =num_features;
    lastObj.rectsLength = (int)indices.size()*num_features;
    lastObj.blurInfo = checkIsBlur(preObjDic, lt, rb);
    lastObj.num_classes = num_classes;
    lastObj.importNetOriImgW = 0;
    lastObj.importNetOriImgH = 0;
    
    clearVector(indices);
    return lastObj;
}


static float cpAverage[10] = {0,0,0,0,0,0,0,0,0,0};
static int cpAveragePointer = 0;

Struct2LastTFLite getTFLiteCenterPoint(void* tflite_datas,void* tflite_datas2,Struct2PreTFLite preObjDic,int tflite_rows2,int tflite_cols2){

    
    Mat mat1 = createLastMatProcess(preObjDic, tflite_datas, NULL, NULL);
    Mat mat2 = createLastMatProcess(preObjDic, tflite_datas2, tflite_rows2, tflite_cols2);

//    判断的分数
//    cout<<mat2.at<float>(0,0)<<endl;
    
    float *rectsArray;
    
    //赋予最新帧的分数
    cpAverage[cpAveragePointer%10] = mat2.at<float>(0,0);
    cpAveragePointer++;
    //取出最近的均值来计算
    float totalResult = 0;
    for(int i = 0;i<10;i++){
        totalResult += cpAverage[i];
    }
    float averageValue = totalResult/10;
    
    //    均值分数
//    cout<<averageValue<<endl;
    
//    if(mat2.at<float>(0,0)>=0.8){
    if(averageValue>=0.95){
        mat1 = backOriImagePosition(preObjDic, mat1);
        if(mat1.at<float>(1,1)>=0){
            float centerLeft = abs(mat1.at<float>(1,0)-mat1.at<float>(0,0))/2+min(mat1.at<float>(1,0),mat1.at<float>(0,0));
            float centerTop = abs(mat1.at<float>(1,1)-mat1.at<float>(0,1))/2+min(mat1.at<float>(1,1),mat1.at<float>(0,1));
            
            Point2f measPt(centerLeft, centerTop);
            Point2f statePt = (centerLeft >= 0 && centerTop >= 0)?kalman.getEstPt(&measPt):kalman.getEstPt(nullptr);
            rectsArray = new float[18]{
                mat1.at<float>(0,0),
                mat1.at<float>(0,1),
                mat1.at<float>(0,0),
                mat1.at<float>(0,1),
                -1,
                100,
                mat1.at<float>(1,0),
                mat1.at<float>(1,1),
                mat1.at<float>(1,0),
                mat1.at<float>(1,1),
                1,
                100,
                statePt.x,
                statePt.y,
                statePt.x,
                statePt.y,
                0,
                100,
            };
        }else{
            rectsArray = new float[18]{0};
        }
    }else{
        rectsArray = new float[18]{0};
    }
    
    Struct2LastTFLite lastObj;
    lastObj.rects = rectsArray;
    lastObj.rectEleCount = 6;
    lastObj.rectsLength = 18;
    return lastObj;
}




int findIndexByVectorElem(vector<const char*> targetVec,const char* targetStr){
    for (int i  =0; i<targetVec.size(); i++) {
        if(targetVec[i] == targetStr){
            return i;
        }
    }
    return -1;
}
    

vector<float> cal_num(vector<vector<vector<float>>> objs,int bound_teeth, int center_teeth,int upper_center,int lower_center,int pos){
    float objs_num = 0;
    float area = 0;
    float center_teeth_x = 0;
    float lower_center_num = 0;
    
//        # process overfit
    if(objs[center_teeth].size() == 0){
        if(center_teeth == upper_center){
            lower_center_num = objs[lower_center].size();
            if(lower_center_num != 0){
                if(lower_center_num > 1){
                    if(pos == 2){
                        center_teeth_x = min(objs[lower_center][0][0], objs[lower_center][1][0]);
                    }
                    if( pos == 4){
                        center_teeth_x = max(objs[lower_center][0][0], objs[lower_center][1][0]);
                    }
                }else{
                    center_teeth_x = objs[lower_center][0][0];
                }
            }else{
                return {objs_num, area};
            }
        }else if(center_teeth == lower_center){
            int upper_center_num = (int)objs[upper_center].size();
            if(upper_center_num != 0){
                if(upper_center_num > 1){
                    if(pos == 2){
                        center_teeth_x = min(objs[upper_center][0][0], objs[upper_center][1][0]);
                    }
                    if( pos == 4){
                        center_teeth_x = max(objs[upper_center][0][0], objs[upper_center][1][0]);
                    }
                }else{
                    center_teeth_x = objs[upper_center][0][0];
                }
            }else{
                return {objs_num, area};
            }
        }
    }else{
        int center_teeth_num = (int)objs[center_teeth].size();
        if(center_teeth_num != 0){
            if(center_teeth_num > 1){
                if(pos == 2){
                    center_teeth_x = min(objs[center_teeth][0][0], objs[center_teeth][1][0]);
                }
                if(pos == 4){
                    center_teeth_x = max(objs[center_teeth][0][0], objs[center_teeth][1][0]);
                }
            }else if( center_teeth_num == 1 ){
                center_teeth_x = objs[center_teeth][0][0];
            }
        }
    }
    for(int label_index = bound_teeth;label_index >= center_teeth-1;label_index--){
        int index_box_num = (int)objs[label_index].size();
        for(int i = 0;i<index_box_num;i++){
            if(pos == 2){
                if(objs[label_index][i][0] - center_teeth_x < 0){
                    objs_num += 1;
                    float w = objs[label_index][i][2] - objs[label_index][i][0];
                    float h = objs[label_index][i][3] - objs[label_index][i][1];
                    area += w * h;
                    break;
                }
            }else if(pos == 4){
                if(objs[label_index][i][0] - center_teeth_x > 0){
                    objs_num += 1;
                    float w = objs[label_index][i][2] - objs[label_index][i][0];
                    float h = objs[label_index][i][3] - objs[label_index][i][1];
                    area += w * h;
                    break;
                }
            }
        }
    }
    return {objs_num, area};
}


vector<float> cal_ul_area(vector<vector<vector<float>>> objs, int index1, int index8){
    float total_area = 0;
    float max_teeth_num = 0;
    for(int i = index8;i>=index1 - 1;i--){
        if(max_teeth_num == 0 and objs[i].size()>0){
            max_teeth_num = i;
        }
        for (int j = 0;j<objs[i].size();j++){
            float w = objs[i][j][2] - objs[i][j][0];
            float h = objs[i][j][3] - objs[i][j][1];
            float area = w * h;
            total_area += area;
        }
    }
    return {total_area, max_teeth_num};

}




vector<float> cal_lr_box_num(vector<vector<vector<float>>> objs, int pos){
//    # upper
    
    int upper_bound =findIndexByVectorElem(labels,"u8");
    int upper_center = findIndexByVectorElem(labels,"u1");
    int lower_bound = findIndexByVectorElem(labels,"l8");
    int lower_center = findIndexByVectorElem(labels,"l1");

//    # if len(objs[lower_center]) == 0:
//    #     lower_center = upper_center
//    # if len(objs[upper_center]) == 0:
//    #     upper_center = lower_center

    vector<float> upper_res = cal_num(objs,upper_bound, upper_center,upper_center,lower_center,pos);
    vector<float> lower_res = cal_num(objs,lower_bound, lower_center,upper_center,lower_center,pos);
    return {upper_res[0] + lower_res[0], upper_res[1] + lower_res[1]};
}



float judge_pos(vector<vector<vector<float>>> objs, int label1,  int label2){
    if(objs[label1].size()==1){
        float u8_cent_x = 0.5 * (objs[label1][0][0] + objs[label1][0][2]);
//        float u8_w = objs[label1][0][2] - objs[label1][0][0];
//        float u8_h = objs[label1][0][3] - objs[label1][0][1];
//        float u8_area = u8_h * u8_w;
        if(objs[label2].size() > 0){
            float u1_cent_x = 0.5 * (objs[label2][0][0] + objs[label2][0][2]);
            if(u1_cent_x - u8_cent_x >= 0){
//                return (4, u8_area)
                return 2;
            }else{
//                return (2, u8_area)
                return 4;
            }
        }
    }
    return -1;
};


vector<float> get_lr_pos(vector<vector<vector<float>>> objs, int label_u1,  int label_l1,  int label_u2,  int label_l2){
    vector<vector<int>> targetLogicVec(4);
    targetLogicVec[0]={label_u2,label_u1};
    targetLogicVec[1]={label_u2,label_l1};
    targetLogicVec[2]={label_l2,label_u1};
    targetLogicVec[3]={label_l2,label_l1};
    
    for(int i=0;i<targetLogicVec.size();i++){
        vector<int> eachTarget = targetLogicVec[i];
        float pos = judge_pos(objs, eachTarget[0], eachTarget[1]);
        if(pos!=-1){
            vector<float> objNumTotalArea = cal_lr_box_num(objs, pos);
            return {pos, objNumTotalArea[0], objNumTotalArea[1]};
        }
    }
    return {};
}


float cal_front_num(vector<vector<vector<float>>> objs, int u1, int u8, int l1, int l8){
float match_num = 0;
for (int i = u1;i<u8 + 1;i++){
    if(objs[i].size() == 2){
        match_num += 1;
    }
}
for (int i = l1;i<l8 + 1;i++){
    if(objs[i].size() == 2){
        match_num += 1;
    }
}
return match_num;
}


vector<float> getDiration(vector<vector<vector<float>>> objs){
    const char* dir_label = "";
    int boxesCount = 0;
    for(int i = 0;i<6;i++){
        if(objs[i].size() >= 1){
            dir_label = labels[i];
            boxesCount ++;
        }
    }
    if(boxesCount!=1){
        return {};
    }else{
        int u1 = findIndexByVectorElem(labels,"u1");
        int l1 = findIndexByVectorElem(labels,"l1");
        int u8 = findIndexByVectorElem(labels,"u8");
        int l8 = findIndexByVectorElem(labels,"l8");
        int u7 = findIndexByVectorElem(labels,"u7");
        int l7 = findIndexByVectorElem(labels,"l7");
        int u6 = findIndexByVectorElem(labels,"u6");
        int l6 = findIndexByVectorElem(labels,"l6");
        
        if(strcmp(dir_label,"lr") ==0 or strcmp(dir_label,"overjet")==0){
            vector<float> pos = get_lr_pos(objs, u1, l1, u8, l8);
            if(pos.size()>0){
                return {pos[0],pos[1],pos[2], 8};
            }
            pos = get_lr_pos(objs, u1, l1, u7, l7);
            if(pos.size()>0){
                return {pos[0],pos[1],pos[2], 7};
            }
            pos = get_lr_pos(objs, u1, l1, u6, l6);
            if(pos.size()>0){
                return {pos[0],pos[1],pos[2], 6};
            }
        }else if (strcmp(dir_label,"upper") == 0 ){
            vector<float> res = cal_ul_area(objs, u1, u8);
            return {1, res[0],res[1]};
        }else if (strcmp(dir_label,"lower") == 0 ){
            vector<float> res = cal_ul_area(objs, l1, l8);
            return {3, res[0],res[1]};
        }else if (strcmp(dir_label,"front") == 0  or strcmp(dir_label,"bite") == 0 ){
            float res = cal_front_num(objs, u1, u8, l1, l8);
            return {res};
        }
    }
    return {};
}
    

typedef struct {
    int leftIndex;
    int rightIndex;
    int upIndex;
    int downIndex;
    int frontIndex;
}Struct2TFLiteMCO;

struct DirUseObj{
    vector<float> dirInfo;
    vector<vector<vector<float>>> objs;
    int index;
};


float dis(float x1, float y1, float x2, float y2){
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

int front_postprocess(vector<DirUseObj> res){
    int u1 = findIndexByVectorElem(labels,"u1");
    int l1 = findIndexByVectorElem(labels,"l1");
    vector<vector<float>> front_res = {};
    for(int i = 0;i<res.size();i++){
        DirUseObj single_res = res[i];
        vector<vector<vector<float>>> objs = single_res.objs;
        if(objs[u1].size() != 2 or objs[l1].size() != 2){
            continue;
        }
        vector<vector<float>> bboxs = {};
        for(int j = 0;j<objs.size();j++){
            vector<vector<float>> obj = objs[j];
            if(obj.size() > 0){
                for (int x = 0; x<obj.size(); x++) {
                    bboxs.push_back(obj[x]);
                }
            }
        }
        sort(bboxs.begin(),bboxs.end(),[](vector<float> o1,vector<float> o2){return o1[0]<o2[0];});
        float rect_xmin = bboxs[0][0];
        sort(bboxs.begin(),bboxs.end(),[](vector<float> o1,vector<float> o2){return o1[1]<o2[1];});
        float rect_ymin = bboxs[0][1];
        sort(bboxs.begin(),bboxs.end(),[](vector<float> o1,vector<float> o2){return o1[2]>o2[2];});
        float rect_xmax = bboxs[0][2];
        sort(bboxs.begin(),bboxs.end(),[](vector<float> o1,vector<float> o2){return o1[3]>o2[3];});
        float rect_ymax = bboxs[0][3];
        float rect_xmid = 0.5 * (rect_xmin + rect_xmax);
        float rect_ymid = 0.5 * (rect_ymin + rect_ymax);
        float dist = 0;
        for(int j = 0;j<objs[u1].size();j++){
            vector<float> obj = objs[u1][j];
            float teeth_xmid = 0.5 * (obj[0] + obj[2]);
            float teeth_ymid = 0.5 * (obj[1] + obj[3]);
            dist += dis(teeth_xmid, teeth_ymid, rect_xmid, rect_ymid);
        }
        for(int j = 0;j<objs[l1].size();j++){
            vector<float> obj = objs[u1][j];
            float teeth_xmid = 0.5 * (obj[0] + obj[2]);
            float teeth_ymid = 0.5 * (obj[1] + obj[3]);
            dist += dis(teeth_xmid, teeth_ymid, rect_xmid, rect_ymid);
        }
        front_res.push_back({dist,(float)single_res.index});
    }
    if(front_res.size() != 0){
        sort(front_res.begin(),front_res.end(),[](vector<float> o1,vector<float> o2){return o1[0]>o2[0];});
        return (int)front_res[0][1];
    }else{
        sort(res.begin(),res.end(),[](DirUseObj o1,DirUseObj o2){return o1.dirInfo[0]>o2.dirInfo[0];});
        return res[0].index;
    }
}
bool sortCmpByUblr(DirUseObj o1,DirUseObj o2){
    if(o1.dirInfo[0]!=o2.dirInfo[0]){
        return o1.dirInfo[0]>o2.dirInfo[0];
    }else{
        if(o1.dirInfo[1]!=o2.dirInfo[1]){
            return o1.dirInfo[1]>o2.dirInfo[1];
        }else{
            if(o1.dirInfo[2]!=o2.dirInfo[2]){
                return o1.dirInfo[2]>o2.dirInfo[2];
            }else{
                return o1.dirInfo[2]<=o2.dirInfo[2];
            }
        }
    }
}

Struct2TFLiteMCO getTFLiteMCO(Struct2LastTFLite lastObj1,Struct2LastTFLite lastObj2,Struct2LastTFLite lastObj3,int objLen){
    vector<DirUseObj> left_res(0);
    vector<DirUseObj> right_res(0);
    vector<DirUseObj> up_res(0);
    vector<DirUseObj> down_res(0);
    vector<DirUseObj> front_res(0);
    
    for (int i = 0; i<objLen; i++) {
        Struct2LastTFLite targetLastObj;
        if(i == 0){
            targetLastObj=lastObj1;
        }else if(i == 1){
            targetLastObj=lastObj2;
        }else if(i == 2){
            targetLastObj=lastObj3;
        }
        
        float* rects = targetLastObj.rects;
        int num_classes = targetLastObj.num_classes;
        vector<vector<vector<float>>> objs(num_classes);
        for (int j = 0; j<targetLastObj.rectsLength/6; j++) {
            float left = rects[j*6];
            float top = rects[j*6+1];
            float right = rects[j*6+2];
            float bottom = rects[j*6+3];
            float index = rects[j*6+5];
            
            objs[index].push_back({left,top,right,bottom});
        }
        vector<float> dirInfo = getDiration(objs);
        
        DirUseObj uobj;
        uobj.dirInfo = dirInfo;
        uobj.objs = objs;
        uobj.index = i;
        
        if(dirInfo.size()>0){
            if(dirInfo.size() > 1){
                if(dirInfo[0] == 2){
                    right_res.push_back(uobj);
                }else if(dirInfo[0] == 4){
                    left_res.push_back(uobj);
                }else if(dirInfo[0] == 1){
                    up_res.push_back(uobj);
                }else if(dirInfo[0] == 3){
                    down_res.push_back(uobj);
                }
            }else{
                front_res.push_back(uobj);
            }
        }
    }
    
    
    int leftIndex = -1;
    if(left_res.size() != 0){
        sort(left_res.begin(),left_res.end(),sortCmpByUblr);
        leftIndex = left_res[0].index;
    }
    int rightIndex = -1;
    if(right_res.size() != 0){
        sort(right_res.begin(),right_res.end(),sortCmpByUblr);
        rightIndex = right_res[0].index;
    }
    int upIndex = -1;
    if(up_res.size() != 0){
        sort(up_res.begin(),up_res.end(),sortCmpByUblr);
        upIndex = up_res[0].index;
    }
    int downIndex = -1;
    if(down_res.size() != 0){
        sort(down_res.begin(),down_res.end(),sortCmpByUblr);
        downIndex = down_res[0].index;
    }
    int frontIndex = -1;
    if(front_res.size() != 0){
        frontIndex = front_postprocess(front_res);
    }
    
    Struct2TFLiteMCO struct2TFLiteMCO;
    struct2TFLiteMCO.leftIndex =leftIndex;
    struct2TFLiteMCO.rightIndex =rightIndex;
    struct2TFLiteMCO.upIndex =upIndex;
    struct2TFLiteMCO.downIndex =downIndex;
    struct2TFLiteMCO.frontIndex =frontIndex;
    return struct2TFLiteMCO;
    
}

typedef struct {
    uint8_t * buffer;
    int bufferLen;
}Struct2AIFace;
Mat gaussianBlur(Mat image){
    int value1 = 2, value2 = 1; //磨皮程度与细节程度的确定
    int dx = value1 * 3;    //双边滤波参数之一
    double fc = value1*10.5; //双边滤波参数之一
    int p = 50; //透明度
    Mat temp1, temp2, temp3, temp4;
    //双边滤波
    bilateralFilter(image, temp1, dx, fc, fc);
    temp2 = (temp1 - image + 128);
    //高斯模糊
    GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
    temp4 = image + 2 * temp3 - 255;
    Mat dst = (image*(100 - p) + temp4*p) / 100;
    //对双边滤波后的图像执行锐化操作，提升图片的棱角以及清晰度
    Mat resultImage;
    Mat kernel = (Mat_<int>(3,3)<< 0, -1, 0, -1, 5, -1, 0, -1, 0);
    filter2D(dst,resultImage,-1,kernel,Point(-1,-1),0);
    return dst;
}



Mat remapTransform(Mat srcMat,int num)
{
    Mat map_x, map_y, result;
    map_x.create(srcMat.size(), CV_32FC1);
    map_y.create(srcMat.size(), CV_32FC1);
    for (int col = 0; col < srcMat.cols; col++) {
        for (int row = 0; row < srcMat.rows; row++) {
            switch (num)
            {
            case -1:///旋转180°
                map_x.at<float>(row, col) = srcMat.cols - col;
                map_y.at<float>(row, col) = srcMat.rows - row;
                break;
            case 0://垂直镜像
                map_x.at<float>(row, col) = col;
                map_y.at<float>(row, col) = srcMat.rows - row;
                break;
            case 1://水平镜像
                map_x.at<float>(row, col) = srcMat.cols - col;
                map_y.at<float>(row, col) = row;
                break;
            }
        }
    }
    remap(srcMat, result, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    map_x.release();
    map_y.release();
    return result;
}


Struct2AIFace getAiFaceImgProcess(Mat oldImg,int isYMirror){
    Mat reslutImgMat = gaussianBlur(oldImg);
    
    //图片是否需要Y轴镜像反转
    if(isYMirror == 1){
        reslutImgMat = remapTransform(reslutImgMat,1);
    }
    
    //转二进制编码
    vector <uint8_t> retv;
    imencode(".jpg", reslutImgMat, retv);
    
    int matTotal =  (int)retv.size();
    uint8_t * uDatas = new uint8_t[matTotal];
    memcpy(uDatas, retv.data(), matTotal);
    
    clearVector(retv);
    
    Struct2AIFace struct2AIFace;
    struct2AIFace.buffer = uDatas;
    struct2AIFace.bufferLen = matTotal;
    
    return struct2AIFace;
}
DART_API Struct2AIFace getAiFaceImgByPath(const char* imgPath,int isYMirror){
    Mat image = imread(imgPath);
    return getAiFaceImgProcess(image,isYMirror);
}
DART_API Struct2AIFace getAiFaceImgByDatas(uint8_t* imgDatas,int width,int height,int isYMirror){
    Mat image = Mat(Size(width,height),CV_8UC3,imgDatas);
    return getAiFaceImgProcess(image,isYMirror);
}


typedef struct {
    bool isUint8;
    float inputInt8TFLiteScale;
    int inputInt8TFLiteZeroPoint;
    float outputInt8TFLiteScale;
    int outputInt8TFLiteZeroPoint;
    int design_width;
    int design_height;
    int outputRows;
    int outputCols;
    int shapeTotalCount;
}StructPredictParams;

static uint8_t* modalOriData;
static TfLiteModel * modal;
static TfLiteInterpreter * interpreter;
static TfLiteInterpreterOptions * interpreterOptions;
static bool isCheckPoint = false;
static StructPredictParams structPredictParams;



void encrypt_one_pass(vector<uint8_t> & plain, const vector<uint8_t> & salt){
    const size_t array_len = plain.size() - 8;
    for(unsigned int i=8; i< array_len + 8; i++){
        unsigned int xor_loc = ((unsigned int)salt[plain[i % 8]] + array_len + i) % 256;
        uint8_t xor_with = salt[xor_loc];
        plain[i] = plain[i] ^ xor_with;
    }
}

void decrypt(vector<uint8_t>& enc_array, const vector<uint8_t> & salt,
                        vector<uint8_t>& result_array){
    encrypt_one_pass(enc_array, salt);
    reverse(enc_array.begin(), enc_array.end());
    encrypt_one_pass(enc_array, salt);
    for(int i = 8;i<enc_array.size();i++){
        result_array.push_back(enc_array[i]);
    }
}


int getTFLiteDecryptResult(uint8_t* bytes,int bytesLen,uint8_t* salt,int saltLen){
    
    vector<uint8_t> vBytes( bytes, bytes + bytesLen);
    vector<uint8_t> vSalt( salt, salt + saltLen);
    vector<uint8_t> decodeList;
    decrypt(vBytes, vSalt, decodeList);
    

    int bufferLen = (int)decodeList.size();
    modalOriData = new uint8_t[bufferLen];
    memcpy(modalOriData, decodeList.data(),bufferLen);
    
    clearVector(vBytes);
    clearVector(vSalt);
    clearVector(decodeList);
    
    return bufferLen;
}


StructPredictParams getStuctPredictParams(){
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    TfLiteType inputType =  TfLiteTensorType(inputTensor);
    TfLiteQuantizationParams inputUint8Params = TfLiteTensorQuantizationParams(inputTensor);
    
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    TfLiteQuantizationParams outputUint8Params = TfLiteTensorQuantizationParams(outputTensor);
    
    bool isUint8 = inputType==kTfLiteUInt8;
    
    StructPredictParams  structPredictParams;
    structPredictParams.isUint8 = isUint8;
    structPredictParams.inputInt8TFLiteScale = inputUint8Params.scale;
    structPredictParams.inputInt8TFLiteZeroPoint = inputUint8Params.zero_point;
    structPredictParams.design_width = TfLiteTensorDim(inputTensor, 2);
    structPredictParams.design_height = TfLiteTensorDim(inputTensor, 1);
    structPredictParams.outputInt8TFLiteScale = outputUint8Params.scale;
    structPredictParams.outputInt8TFLiteZeroPoint = outputUint8Params.zero_point;
    structPredictParams.outputRows = TfLiteTensorDim(outputTensor, 2);
    structPredictParams.outputCols = TfLiteTensorDim(outputTensor, 1);
    return structPredictParams;
}


static bool tfliteRuning = false;
static bool tfliteIsWaitFree = false;
DART_API void deleteTFLite(){
    if(interpreter != nullptr){
        if(!tfliteRuning){
            tfliteIsWaitFree = false;
            
            kalman.resetKalman();
            
            TfLiteInterpreterDelete(interpreter);
            TfLiteInterpreterOptionsDelete(interpreterOptions);
            TfLiteModelDelete(modal);
            
            interpreter = NULL;
            interpreterOptions = NULL;
            modal = NULL;
            
            delete [] modalOriData;
            modalOriData = NULL;
        }else{
            tfliteIsWaitFree = true;
        }
    }
}


DART_API void createTFLite(uint8_t* modalData,int modalDataLen,int isEncrypt,int isCheckTwoPoint){
    if(isCheckTwoPoint){
        kalman.resetKalman();
    }
    if(interpreter != nullptr){
        deleteTFLite();
    }
    
    isCheckPoint = isCheckTwoPoint;
    
    if(isEncrypt){
        int saltLen = 256;
        uint8_t * salt = new uint8_t[saltLen]{88,1, 109, 253, 126, 113, 241, 98, 204, 16, 220, 27, 235, 122, 74, 229, 165, 219, 117, 34, 185, 4, 12, 71, 150, 245, 183, 11, 21, 219, 47, 93, 209, 48, 10, 147, 195, 112, 105, 57, 230, 170, 132, 197, 199, 189, 127, 164, 85, 64, 78, 65, 201, 167, 32, 147, 96, 254, 178, 100, 225, 37, 109, 185, 185, 44, 42, 248, 117, 255, 180, 94, 230, 167, 148, 192, 196, 131, 71, 247, 13, 208, 155, 76, 216, 3, 197, 222, 146, 169, 202, 91, 249, 69, 187, 250, 117, 86, 183, 124, 223, 65, 8, 128, 181, 92, 209, 186, 73, 219, 236, 235, 130, 175, 20, 143, 80, 200, 157, 124, 12, 70, 194, 73, 140, 22, 215, 184, 37, 129, 22, 12, 170, 39, 0, 32, 78, 247, 19, 52, 100, 234, 89, 44, 158, 237, 82, 112, 24, 154, 62, 203, 217, 153, 1, 11, 239, 183, 245, 11, 190, 52, 210, 208, 111, 109, 48, 79, 107, 220, 201, 86, 133, 18, 174, 4, 121, 214, 8, 3, 228, 230, 62, 33, 170, 201, 153, 78, 157, 47, 150, 69, 113, 154, 116, 198, 238, 198, 113, 134, 223, 122, 218, 33, 171, 230, 65, 25, 208, 248, 199, 169, 89, 117, 138, 5, 239, 120, 188, 127, 174, 57, 87, 19, 160, 127, 38, 184, 12, 23, 239, 23, 230, 150, 42, 175, 10, 55, 165, 46, 134, 107, 130, 44, 59, 31, 151, 30, 80, 153, 70, 85, 38, 114, 205, 51};
        
        //ffi 用完就清理了，这里保留一份 modalOriData ，保证interpreter源不会丢失出现EXC_BAD_ACCESS
        int decryptLen =  getTFLiteDecryptResult(modalData,modalDataLen,salt,saltLen);
        modal = TfLiteModelCreate(modalOriData, decryptLen);

        delete [] salt;
        salt = NULL;
    }else{
        //ffi 用完就清理了，这里保留一份 modalOriData ，保证interpreter源不会丢失出现EXC_BAD_ACCESS
        modalOriData = new uint8_t[modalDataLen];
        memcpy(modalOriData, modalData, modalDataLen);
        modal = TfLiteModelCreate(modalOriData, modalDataLen);
    }
    interpreterOptions = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(interpreterOptions, 4);
    
    interpreter = TfLiteInterpreterCreate(modal, interpreterOptions);
    TfLiteInterpreterAllocateTensors(interpreter);
    
    structPredictParams = getStuctPredictParams();
}

typedef struct{
    Struct2LastTFLite last1;
    Struct2LastTFLite last2;
    Struct2LastTFLite last3;
    Struct2TFLiteMCO mco;
    int runMs;
    int totalMs;
} Struct2PredictResult;
Struct2PredictResult predictTFLite(vector<Struct2PreTFLite> &vectorPreTFLites,long startPreMS,bool isNewModal){
    
    bool isUint8 = vectorPreTFLites[0].isInt8TFLite;
    
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    
    int inputBytesLen = (int)TfLiteTensorByteSize(inputTensor);
    int eachInputImgBytesSize = inputBytesLen/vectorPreTFLites.size();
    
    if(isUint8){
        uint8_t* inputBuffer = new uint8_t[inputBytesLen];
        for (int i = 0; i<vectorPreTFLites.size(); i++) {
            Struct2PreTFLite each = vectorPreTFLites[i];
            //使用 eachInputImgBytesSize 保证模型所需字节数满足，比如：模型需求3张图，但是你传入了一张图进行检测，那么申请3张图内存，进行1张图检测也支持
            memcpy(&inputBuffer[i*eachInputImgBytesSize], each.preImgBytes_uint8, each.preImgBytesLen*sizeof(uint8_t));

            delete [] each.preImgBytes_uint8;
            each.preImgBytes_uint8 = NULL;
        }
        TfLiteTensorCopyFromBuffer(inputTensor, inputBuffer, inputBytesLen);

        delete [] inputBuffer;
        inputBuffer = NULL;
    }else{
        float* inputBuffer = new float[inputBytesLen];
        for (int i = 0; i<vectorPreTFLites.size(); i++) {
            Struct2PreTFLite each = vectorPreTFLites[i];
            //使用 eachInputImgBytesSize 保证模型所需字节数满足，比如：模型需求3张图，但是你传入了一张图进行检测，那么申请3张图内存，进行1张图检测也支持
            memcpy(&inputBuffer[i*eachInputImgBytesSize], each.preImgBytes_float, each.preImgBytesLen*sizeof(float));

            delete [] each.preImgBytes_float;
            each.preImgBytes_float = NULL;
        }
        TfLiteTensorCopyFromBuffer(inputTensor, inputBuffer, inputBytesLen);
        delete [] inputBuffer;
        inputBuffer = NULL;
    }
    int runMs = 0;
    try{
        long runMsStartMS = getTimestamp();
        TfLiteInterpreterInvoke(interpreter);
        runMs = (int)(getTimestamp()-runMsStartMS);
    }catch (exception& e){
        cout << "推测异常 exception: " << e.what() << endl;
    }
    
    
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    const TfLiteTensor* outputTensor2 = TfLiteInterpreterGetOutputTensor(interpreter, 1);
    
    int output1BytesLen = (int)TfLiteTensorByteSize(outputTensor);
    
    void* outputBuffer = nullptr;
    void* outputBuffer2 = nullptr;
    
    Struct2PredictResult struct2PredictResult;
    if(isUint8){
        outputBuffer = new uint8_t[output1BytesLen];
        TfLiteTensorCopyToBuffer(outputTensor, outputBuffer, output1BytesLen);
        
        if(TfLiteInterpreterGetOutputTensorCount(interpreter)>1){
            int output2BytesLen = (int)TfLiteTensorByteSize(outputTensor2);
            outputBuffer2 = new uint8_t[output2BytesLen/sizeof(uint8_t)];
            TfLiteTensorCopyToBuffer(outputTensor2, outputBuffer2, output2BytesLen);
        }
    }else{
        outputBuffer = new float[output1BytesLen];
        TfLiteTensorCopyToBuffer(outputTensor, outputBuffer, output1BytesLen);
        if(TfLiteInterpreterGetOutputTensorCount(interpreter)>1){
            int output2BytesLen = (int)TfLiteTensorByteSize(outputTensor2);
            outputBuffer2 = new float[output2BytesLen/sizeof(float)];
            TfLiteTensorCopyToBuffer(outputTensor2, outputBuffer2, output2BytesLen);
        }
    }
    for (int i = 0; i<vectorPreTFLites.size(); i++) {
        Struct2PreTFLite each = vectorPreTFLites[i];
        Struct2LastTFLite lastTFLite;
        if(isCheckPoint){
            lastTFLite = getTFLiteCenterPoint(outputBuffer, outputBuffer2, each, TfLiteTensorDim(outputTensor2, 1), TfLiteTensorDim(outputTensor2, 0));
        }else{
            int eachOneTotalBytesSize = output1BytesLen/vectorPreTFLites.size();
            if(isUint8){
                uint8_t *eachBuffer = new uint8_t[eachOneTotalBytesSize];
                memcpy(eachBuffer, ((uint8_t*)outputBuffer)+(i*eachOneTotalBytesSize), eachOneTotalBytesSize);
                if(isNewModal){
                    lastTFLite = newLastTFLiteProcess(eachBuffer, each);
                }else{
                    lastTFLite = lastTFLiteProcess(eachBuffer, each);
                }
                delete [] eachBuffer;
                eachBuffer = NULL;
            }else{
                float *eachBuffer = new float[eachOneTotalBytesSize];
                memcpy(eachBuffer, ((float*)outputBuffer)+(i*eachOneTotalBytesSize), eachOneTotalBytesSize);
                if(isNewModal){
                    lastTFLite = newLastTFLiteProcess(eachBuffer, each);
                }else{
                    lastTFLite = lastTFLiteProcess(eachBuffer, each);
                }
                delete [] eachBuffer;
                eachBuffer = NULL;
            }
        }
        
        delete [] each.imgDataList;
        each.imgDataList= NULL;
        
        lastTFLite.importNetOriImgW = each.ori_size_w;
        lastTFLite.importNetOriImgH = each.ori_size_h;
        if(i == 0){
            struct2PredictResult.last1 = lastTFLite;
        }else if(i == 1){
            struct2PredictResult.last2 = lastTFLite;
        }else if(i == 2){
            struct2PredictResult.last3 = lastTFLite;
        }
    }
    
    if(!isCheckPoint && !isNewModal){
        struct2PredictResult.mco = getTFLiteMCO(struct2PredictResult.last1, struct2PredictResult.last2, struct2PredictResult.last3, (int)vectorPreTFLites.size());
    }
    
    //free
    if(isUint8){
        if(outputBuffer!=nullptr){
            delete static_cast<uint8_t*>(outputBuffer);
            outputBuffer = NULL;
        }
        if(outputBuffer2!=nullptr){
            delete static_cast<uint8_t*>(outputBuffer2);
            outputBuffer2 = NULL;
        }
    }else{
        if(outputBuffer!=nullptr){
            delete static_cast<float*>(outputBuffer);
            outputBuffer = NULL;
        }
        if(outputBuffer2!=nullptr){
            delete static_cast<float*>(outputBuffer2);
            outputBuffer2 = NULL;
        }
    }
    
    tfliteRuning = false;
    if(tfliteIsWaitFree){
        deleteTFLite();
    }
    clearVector(vectorPreTFLites);
    
    //
    struct2PredictResult.runMs = runMs;
    struct2PredictResult.totalMs = (int)(getTimestamp() - startPreMS);
    return struct2PredictResult;
}



DART_API Struct2PredictResult predictByPaths(const char* imgPath,const char* imgPath2,const char* imgPath3,int pathLen,int isUseNewModal){
    if(interpreter == NULL || tfliteIsWaitFree){
        Struct2PredictResult result;
        return result;
    }
    
    tfliteRuning = true;
    
    long startPreMS = getTimestamp();
    vector<Struct2PreTFLite> preTflitesVec(0);
    for (int i = 0; i<pathLen; i++) {
        const char* targetPath = "";
        if(i == 0){
            targetPath = imgPath;
        }else if(i == 1){
            targetPath = imgPath2;
        }else if(i == 2){
            targetPath = imgPath3;
        }
        Struct2PreTFLite preLite = preTFLiteProcess(nullptr, 0, 0,targetPath, structPredictParams.isUint8, structPredictParams.inputInt8TFLiteScale, structPredictParams.inputInt8TFLiteZeroPoint, structPredictParams.design_width, structPredictParams.design_height, isCheckPoint,0,nullptr,nullptr,0,0,0,isUseNewModal == 1);
        preLite.imgPath = targetPath;
        preLite.rows = structPredictParams.outputRows;
        preLite.cols = structPredictParams.outputCols;
        preLite.outputInt8TFLiteScale = structPredictParams.outputInt8TFLiteScale;
        preLite.outputInt8TFLiteZeroPoint = structPredictParams.outputInt8TFLiteZeroPoint;
        preTflitesVec.push_back(preLite);
    }
    return predictTFLite(preTflitesVec,startPreMS,isUseNewModal == 1);
}

//imgDatasType:
//0 = rgb 格式
//1 = rbga8888 格式
//2 = yuv420 格式
DART_API Struct2PredictResult predictByData(uint8_t * imgData,int imgDataLen,int width,int height,int imgDatasType,uint8_t* plane1,uint8_t* plane2, int bytesPerRow, int bytesPerPixel, int yRowStride, int isUseNewModal){
    
    if(interpreter == NULL || tfliteIsWaitFree){
        Struct2PredictResult result;
        return result;
    }
    
    tfliteRuning = true;
    
    long startPreMS = getTimestamp();
    vector<Struct2PreTFLite> preTflitesVec(0);
    Struct2PreTFLite preLite = preTFLiteProcess(imgData, width, height,"", structPredictParams.isUint8, structPredictParams.inputInt8TFLiteScale, structPredictParams.inputInt8TFLiteZeroPoint, structPredictParams.design_width, structPredictParams.design_height, isCheckPoint,imgDatasType,plane1,plane2,bytesPerRow,bytesPerPixel,yRowStride,isUseNewModal == 1);
    preLite.imgPath = "";
    preLite.rows = structPredictParams.outputRows;
    preLite.cols = structPredictParams.outputCols;
    preLite.outputInt8TFLiteScale = structPredictParams.outputInt8TFLiteScale;
    preLite.outputInt8TFLiteZeroPoint = structPredictParams.outputInt8TFLiteZeroPoint;
    
    preTflitesVec.push_back(preLite);
    
    return predictTFLite(preTflitesVec,startPreMS,isUseNewModal == 1);
}


//quality: [0,100]
DART_API void compressImg(const char* imgPath,const char* imgOutPath,int tWidth,int tHeight,int quality){
    Mat oldImgMat = imread(imgPath);
    Mat newImgMat;
    if(tWidth != 0 || tHeight !=0){
        if(tWidth !=0 && tHeight !=0){
            resize(oldImgMat, newImgMat, Size(tWidth,tHeight));
        }else{
            Size newSize;
            if(tWidth == 0){
                newSize =Size(tHeight/oldImgMat.rows*oldImgMat.cols,tHeight);
            }else{
                newSize =Size(tWidth, float(tWidth) / oldImgMat.cols * oldImgMat.rows);
            }
            resize(oldImgMat, newImgMat, newSize);
        }
    }
    vector <int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(quality>=100?100:quality);
    imwrite(imgOutPath, (tWidth != 0 || tHeight !=0)?newImgMat:oldImgMat,compression_params);
    clearVector(compression_params);
}

typedef struct{
    int firstValue;
    int secondValue;
} Struct2CheckLightResult;
Struct2CheckLightResult checkLightProcess(Mat img){
    Mat mask = Mat::ones(img.size(), CV_8UC1);
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat binary;
    threshold(gray, binary, 127, 255, THRESH_BINARY);
    
    vector<vector<Point>> contours = {};
    findContours(binary, contours,RETR_LIST, CHAIN_APPROX_SIMPLE);
    float maxLenValue = 0;
    float maxLenIndex = 0;
    for (int i = 0;i<contours.size();i++ )
    {
        if(contours[i].size() >= maxLenValue){
            maxLenValue = contours[i].size();
            maxLenIndex = i;
        }
    }
    
    Struct2CheckLightResult struct2CheckLightResult;
    struct2CheckLightResult.firstValue = 0;
    struct2CheckLightResult.secondValue = 0;
    
    if(contours.size()>0){
        //第一个值
        fillPoly(mask, contours[maxLenIndex], Scalar(0,0,0));
        
        Scalar resultScalar = mean(img,mask==1);
        float result = (resultScalar[0]+resultScalar[1]+resultScalar[2])/3;
//        cout<<resultScalar<<endl;
//        cout<<result<<endl;
        clearVector(contours);
        struct2CheckLightResult.firstValue = (int)result;
        //第二个值(方案不稳定，受到开口器位置影响较大)
//        Mat t1Mat = img(Range::all(),Range(img.cols - img.cols/10,img.cols));
//        Mat t1MatSort =  Mat::zeros(t1Mat.rows, t1Mat.cols, t1Mat.type());
//        //col 倒序
//        for( int nrow = 0; nrow < t1Mat.rows; nrow++)
//        {
//            for(int ncol = 0; ncol < t1Mat.cols; ncol++)
//            {
//                t1MatSort.at<Vec3b>(nrow,ncol) = t1Mat.at<Vec3b>(nrow,t1Mat.cols - 1 - ncol);
//            }
//        }
//        Mat t2Mat = img(Range::all(),Range(0,img.cols/10));
//        //防止减出来的数值是负数直接变成0,转换类型为 CV_32FC3
//        Mat t1ConverMat;
//        Mat t2ConverMat;
//        t1MatSort.convertTo(t1ConverMat, CV_32FC3);
//        t2Mat.convertTo(t2ConverMat, CV_32FC3);
//        Mat t3Mat = t1ConverMat - t2ConverMat;
//        Scalar result2Scalar = mean(t3Mat.mul(t3Mat));
//        float result2 = (result2Scalar[0]+result2Scalar[1]+result2Scalar[2])/3;
//        struct2CheckLightResult.secondValue = (int)result2;
    }
    
    return struct2CheckLightResult;
}


DART_API Struct2CheckLightResult checkImgHaveLightByData(uint8_t* imgDatas,int width,int height){
    Mat img = Mat(Size(width,height),CV_8UC3,imgDatas);
    return checkLightProcess(img);
}

DART_API Struct2CheckLightResult checkImgHaveLight(const char* imgPath){
    Mat img = imread(imgPath);
    return checkLightProcess(img);
}


int checkToothLightProcess(Mat img,int centerX,int centerY){
//    Mat mask = Mat::zeros(img.size(), CV_8UC1);
//    Mat gray;
//    cvtColor(img, gray, COLOR_BGR2GRAY);
//    Mat binary;
//    threshold(gray, binary, 127, 255, THRESH_BINARY);
//
//    vector<vector<Point>> contours = {};
//    findContours(binary, contours,RETR_LIST, CHAIN_APPROX_SIMPLE);
//    float maxLenValue = 0;
//    float maxLenIndex = 0;
//    for (int i = 0;i<contours.size();i++ )
//    {
//        if(contours[i].size() >= maxLenValue){
//            maxLenValue = contours[i].size();
//            maxLenIndex = i;
//        }
//    }
//
//    if(contours.size()>0){
//        fillPoly(mask, contours[maxLenIndex], Scalar(1,1,1));
//
//        //之前的方案
////        Scalar resultScalar = mean(img,mask==1);
////        float result = (resultScalar[0]+resultScalar[1]+resultScalar[2])/3;
//        //新方案
//        float result = 0;
//        for( int nrow = 0; nrow < mask.rows; nrow++)
//        {
//           for(int ncol = 0; ncol < mask.cols; ncol++)
//           {
//               if(mask.at<Vec3b>(nrow,ncol)[0] == 1 || mask.at<Vec3b>(nrow,ncol)[1] == 1 || mask.at<Vec3b>(nrow,ncol)[2] == 1){
//                   if(img.at<Vec3b>(nrow,ncol)[0] >= 200 || img.at<Vec3b>(nrow,ncol)[1] >= 200 || img.at<Vec3b>(nrow,ncol)[2] >= 200){
//                       result++;
//                   }
//               }
//           }
//        }
//
//        clearVector(contours);
//        return (int)result;
//    }else{
//        return 0;
//    }
    
    //最新方案
    int rectWH = img.rows/3;
    int startRowIndex =centerY-rectWH/2>=0?centerY-rectWH/2:0;
    int endRowIndex = centerY+rectWH/2;
    int startColIndex = centerX-rectWH/2>=0?centerX-rectWH/2:0;
    int endColIndex = centerX+rectWH/2;
    
//    Mat rectMat = img(Range(startRowIndex,endRowIndex),Range(startColIndex,endColIndex));
//    Scalar scalarMean =  mean(rectMat);
//    return (scalarMean[0]+scalarMean[1]+scalarMean[2])/3;
    
    float allSumValue = 0;
    int allSumCounter = 0;
    for( int nrow = startRowIndex; nrow < endRowIndex; nrow++)
    {
       for(int ncol = startColIndex; ncol < endColIndex; ncol++)
       {
           allSumCounter ++;
           Vec3i channels =img.at<Vec3b>(nrow,ncol)[1];
           if(channels[0]>=channels[1] && channels[0]>=channels[2]){
               allSumValue += channels[0];
           }else if(channels[1]>=channels[0] && channels[1]>=channels[2]){
               allSumValue += channels[1];
           }else if(channels[2]>=channels[0] && channels[2]>=channels[1]){
               allSumValue += channels[2];
           }
       }
    }
    return (int)(allSumValue/allSumCounter);
}

DART_API int checkToothHaveLight(const char* imgPath,int centerX,int centerY){
    Mat img = imread(imgPath);
    return checkToothLightProcess(img,centerX,centerY);
}
DART_API int checkToothHaveLightByData(uint8_t* imgDatas,int width,int height,int centerX,int centerY){
    Mat img = Mat(Size(width,height),CV_8UC3,imgDatas);
    return checkToothLightProcess(img,centerX,centerY);
}



DART_API void colorTransform(const char* imgPath){
    Mat img = imread(imgPath);
    Mat result;
    cvtColor(img,result,COLOR_BGR2Lab);
    
    Scalar meanScalar = mean(result);
    float avg_a = meanScalar.val[1];
    float avg_b = meanScalar.val[2];
    float tempMaxA = (avg_a-144)>0?(avg_a-144)/16:0;
    float tempMaxB = (avg_b-144)>0?(avg_b-144)/16:0;
    float coefficient1 = tempMaxA>=0.8?0.8:tempMaxA;
    float coefficient2 = tempMaxB>=1.1?1.1:tempMaxB;
     
    
    for( int nrow = 0; nrow < result.rows; nrow++)
    {
       for(int ncol = 0; ncol < result.cols; ncol++)
       {
           result.at<Vec3b>(nrow,ncol)[1] = result.at<Vec3b>(nrow,ncol)[1] - ((avg_a - 128) * (result.at<Vec3b>(nrow,ncol)[0]/255.0)*coefficient1);
           result.at<Vec3b>(nrow,ncol)[2] = result.at<Vec3b>(nrow,ncol)[2] - ((avg_b - 128) * (result.at<Vec3b>(nrow,ncol)[0]/255.0)*coefficient2);
       }
    }
    
    Mat processDoneMat;
    cvtColor(result,processDoneMat,COLOR_Lab2BGR);
    
    
    vector <int> compression_params;
    compression_params.push_back(IMWRITE_JPEG_QUALITY);
    compression_params.push_back(100);
    imwrite(imgPath, processDoneMat,compression_params);
    clearVector(compression_params);
    
}
