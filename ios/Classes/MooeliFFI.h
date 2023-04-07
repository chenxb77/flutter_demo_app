//
// Created by Himi on 2022/8/3.
//

#include <stdint.h>
typedef struct {
    uint8_t * buffer;
    int bufferLen;
}Struct2AIFace;
Struct2AIFace getAiFaceImgByPath(const char* imgPath,int isYMirror);
Struct2AIFace getAiFaceImgByDatas(uint8_t* imgDatas,int width,int height,int isYMirror);

//
typedef struct {
    int leftIndex;
    int rightIndex;
    int upIndex;
    int downIndex;
    int frontIndex;
}Struct2TFLiteMCO;
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
typedef struct{
    Struct2LastTFLite last1;
    Struct2LastTFLite last2;
    Struct2LastTFLite last3;
    Struct2TFLiteMCO mco;
    int runMs;
    int totalMs;
} Struct2PredictResult;
void createTFLite(uint8_t* modalData,int modalDataLen,int isEncrypt,int isCheckTwoPoint);
void deleteTFLite();
Struct2PredictResult predictByPaths(const char* imgPath,const char* imgPath2,const char* imgPath3,int pathLen,int isUseNewModal);
Struct2PredictResult predictByData(uint8_t * imgData,int imgDataLen,int width,int height,int imgDatasType,uint8_t* plane1,uint8_t* plane2, int bytesPerRow, int bytesPerPixel, int yRowStride, int isUseNewModal);

//quality: [0,100]
void compressImg(const char* imgPath,const char* imgOutPath,int tWidth,int tHeight,int quality);

typedef struct{
    int firstValue;
    int secondValue;
} Struct2CheckLightResult;
Struct2CheckLightResult checkImgHaveLight(const char* imgPath);
Struct2CheckLightResult checkImgHaveLightByData(uint8_t* imgDatas,int width,int height);


int checkToothHaveLight(const char* imgPath,int centerX,int centerY);
int checkToothHaveLightByData(uint8_t* imgDatas,int width,int height,int centerX,int centerY);

//针对偏黄色调的图进行处理
void colorTransform(const char* imgPath);

