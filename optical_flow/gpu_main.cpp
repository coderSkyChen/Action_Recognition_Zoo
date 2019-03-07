#include<sys/types.h>
#include<sys/stat.h>
#include <iostream>
#include <fstream>
#include<dirent.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include"opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace cv::gpu;
using namespace std;


bool clipFlow = true;

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.f;
        const float col1 = colorWheel[k1][b] / 255.f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.f * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion = -1)
{
    dst.create(flow.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y)
        {
            for (int x = 0; x < flow.cols; ++x)
            {
                Point2f u = flow(y, x);

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            Point2f u = flow(y, x);

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

// binary file format for flow data specified here:
// http://vision.middlebury.edu/flow/data/
static void writeOpticalFlowToFile(const Mat_<Point2f>& flow, const string& fileName)
{
    static const char FLO_TAG_STRING[] = "PIEH";

    ofstream file(fileName.c_str(), ios_base::binary);

    file << FLO_TAG_STRING;

    file.write((const char*) &flow.cols, sizeof(int));
    file.write((const char*) &flow.rows, sizeof(int));

    for (int i = 0; i < flow.rows; ++i)
    {
        for (int j = 0; j < flow.cols; ++j)
        {
            const Point2f u = flow(i, j);

            file.write((const char*) &u.x, sizeof(float));
            file.write((const char*) &u.y, sizeof(float));
        }
    }
}

static void convertFlowToImage(const Mat &flow_x, Mat &img_x, double lowerBound, double higherBound)
{
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_x.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
        }
    }
    #undef CAST
}



int main(int argc, const char* argv[])
{
  //  if (argc < 3)
  //  {
  //      cerr << "Usage : " << argv[0] << "<frame0> <frame1> [<output_flow>]" << endl;
  //      return -1;
  //  }

    Mat image,prev_image,flow_x,flow_y;

    GpuMat frame_0, frame_1, flow_u, flow_v;

    setDevice(0);
    OpticalFlowDual_TVL1_GPU alg_tvl1;

    string outputPath="/home/mcg/cxk/dataset/somthing-something/something-optical-flow/";
    DIR* dir=opendir("/home/mcg/cxk/dataset/somthing-something/something-rgb");
    dirent* p=NULL;
    while((p=readdir(dir))!=NULL)
    {
        if(p->d_name[0]!='.')
        {
            string name="/home/mcg/cxk/dataset/somthing-something/something-rgb/"+string(p->d_name);
            cout<<name<<endl;

            int status;
            string newPath=outputPath+string(p->d_name);
            status=mkdir(newPath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if(status!=0)
            {
                cout<<"error in mkdir!"<<endl;
                cout<<status<<endl;
                cout<<newPath<<endl;
                return -1;
            }


            int count=0;
            DIR* dirChild=opendir(name.c_str());
            dirent* pChild=NULL;
            while((pChild=readdir(dirChild))!=NULL)
            {
                if(pChild->d_name[0]!='.')
                {
                    string nameChild=name+'/'+string(pChild->d_name);
                    //cout<<nameChild<<endl;
                    ++count;
                }
            }
            closedir(dirChild);
            
            for(int i=1;i<=count;i+=2)
            {
                char tmp[20];
                sprintf(tmp,"%05d.jpg",int(i));
                string imgName=name+'/'+tmp;
                if(i==1)
                {
                    prev_image = imread(imgName, IMREAD_GRAYSCALE);
                    if(prev_image.empty())
                    {
                        cerr << "Can't open image ["  << imgName << "]" << endl;
                        return -1;
                    }

                }
                image=imread(imgName, IMREAD_GRAYSCALE);
                if(image.empty())
                {
                    cerr << "Can't open image ["  << imgName << "]" << endl;
                    return -1;
                }
                if (image.size() != prev_image.size())
                {
                    cerr << "Images should be of equal sizes" << endl;
                    cerr<<image.size()<<prev_image.size()<<endl;
                    return -1;
                }

                frame_0.upload(prev_image);
                frame_1.upload(image);

                alg_tvl1(frame_0,frame_1,flow_u,flow_v);

                flow_u.download(flow_x);
                flow_v.download(flow_y);

                Mat imgX(flow_x.size(),CV_8UC1);
                Mat imgY(flow_y.size(),CV_8UC1);


                double min_x,max_x;
                double min_y,max_y;
                minMaxLoc(flow_x,&min_x,&min_x);
                minMaxLoc(flow_y,&min_y,&min_y);

                if(clipFlow)
                {
                    min_x=-20; max_x=20;
                    min_y=-20; max_y=20;
                }
                convertFlowToImage(flow_x,imgX,min_x,max_x);
                convertFlowToImage(flow_y,imgY,min_y,max_y);

                imwrite(outputPath+string(p->d_name)+'/'+"x_"+tmp,imgX);
                imwrite(outputPath+string(p->d_name)+'/'+"y_"+tmp,imgY);

                std::swap(prev_image,image);
                           
            }

        }
    }
    closedir(dir);

//    Mat out;
//    drawOpticalFlow(flow, out);
//
//    if (argc == 4)
//        writeOpticalFlowToFile(flow, argv[3]);
//
//    imwrite("Flow.jpg", out);
//    cout<<"ok"<<endl;
//
//    waitKey();

    return 0;
}
