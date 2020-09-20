// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/imgproc/types_c.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat MoveDetect(Mat background, Mat img,rs2::depth_frame depth)
{   
	//将background和img转为灰度图
	Mat result = img.clone();
	Mat gray1, gray2;
	cvtColor(background, gray1, CV_BGR2GRAY);
	cvtColor(img, gray2, CV_BGR2GRAY);
 	
//进行canny边缘检测
        //Canny(gray1, gray1, 0, 30, 3);
 
	//将background和img做差；对差值图diff进行阈值化处理
	Mat diff;
	absdiff(gray1, gray2, diff);
	imshow("absdiss", diff);
	threshold(diff, diff, 30, 255, THRESH_BINARY);
	imshow("threshold", diff);
 
	//腐蚀膨胀消除噪音
	
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(15, 15));
	//erode(diff, diff, element);
	//imshow("erode", diff);
	dilate(diff, diff, element2);
	imshow("dilate", diff);
	
 
	//二值化后使用中值滤波+膨胀
	//Mat element = getStructuringElement(MORPH_RECT, Size(11, 11));
	medianBlur(diff, diff, 5);//中值滤波
	imshow("medianBlur", diff);
	//dilate(diff, diff, element);
	//blur(diff, diff, Size(10, 10)); //均值滤波
	imshow("dilate", diff);
 
	//查找并绘制轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarcy;
	findContours(diff, contours, hierarcy, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE); //查找轮廓
	vector<Rect> boundRect(contours.size()); //定义外接矩形集合
	//drawContours(img2, contours, -1, Scalar(0, 0, 255), 1, 8);  //绘制轮廓
 
	//查找正外接矩形
	int x0 = 0, y0 = 0, w0 = 0, h0 = 0;
	double Area = 0  ,  AreaAll = 0  ;
        float dist_to_center;
	for (int i = 0; i<contours.size(); i++)
	{
		boundRect[i] = boundingRect((Mat)contours[i]); //查找每个轮廓的外接矩形
		x0 = boundRect[i].x;  //获得第i个外接矩形的左上角的x坐标
		y0 = boundRect[i].y; //获得第i个外接矩形的左上角的y坐标
		w0 = boundRect[i].width; //获得第i个外接矩形的宽度
		h0 = boundRect[i].height; //获得第i个外接矩形的高度
 		cout<<"x0"<<x0<<" y0"<<y0<<endl;
		//计算面积
		double Area = contourArea(contours[i]);//计算第i个轮廓的面积
		AreaAll = Area + AreaAll;
		
		//筛选
		if (w0>10 && h0>10 && w0<300 && h0<300)
                {
                float c_depth=999;
                for (int j=-100;j<100;j++)
                {
                    int xj=x0+(int)(0.5*w0)+j;
                    int yj=yj+(int)(0.5*h0)+j;
                    if (xj<1)xj=1;
                    if(xj>=640)xj=639;
                    if(yj<1)yj=1;
                    if(yj>=480)yj=479;
		    dist_to_center = depth.get_distance(xj, yj);
                    if (dist_to_center >=0.1)
                    {
                        if(dist_to_center<c_depth)c_depth=dist_to_center;
                    }
                    //cout<<"w"<<w0<<"h"<<h0<<endl;
                    //cout<<"depth"<<dist_to_center<<endl;
                    //cout<<"ccccccccccccccccccc_depth"<<c_depth<<endl;
		}
                rectangle(result, Point(x0, y0), Point(x0 + w0, y0 + h0), Scalar(0, 255, 0), 2, 8); //绘制第i个外接矩形
                } 
		//文字输出
		Point org(10, 35);
		if (i >= 1 && AreaAll>=19600)
		putText(result, "Is Blocked ", org , FONT_HERSHEY_SIMPLEX,0.8f,Scalar(0, 255, 0),2);
 
	}
	return result;
}

int main(int argc, char * argv[])try
{
    //Contruct a pipeline which abstracts the device
    rs2::pipeline p;

    //Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    //Add desired streams to configuration
    //cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);

    //Instruct pipeline to start streaming with the requested configuration
    //p.start(cfg);
    p.start();

    
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    namedWindow("mask", WINDOW_AUTOSIZE );
    namedWindow("diff", WINDOW_AUTOSIZE );
    //namedWindow("test",CV_WINDOW_AUTOSIZE);

    Mat frame,mask,threImage,output;
    //int delay = 1000/cam.get(CV_CAP_PROP_FPS);
    //BackgroundSubtractorMOG bgSubtractor(10,10,0.5,false);
    //构造混合高斯模型 参数1：使用历史帧的数量 2：混合高斯个数，3：背景比例 4：:噪声权重
    Mat framePro,dframe;

    Mat background;
    Mat result;
    int count = 0;

    bool flag = false;

    while(1)
    {
        // Camera warmup - dropping several first frames to let auto-exposure stabilize
        rs2::frameset frames;
        //for(int i = 0; i < 30; i++)
        //{
            //Wait for all configured streams to produce a frame
        frames = p.wait_for_frames();
        //}

        //Get each frame
        rs2::frame color_frame = frames.get_color_frame();
        // Try to get a frame of a depth image
        rs2::depth_frame depth = frames.get_depth_frame();
        //float dist_to_center;

        //dist_to_center = depth.get_distance(20, 30);       
        //cout<<"depth"<<dist_to_center<<endl;

	// Get the depth frame's dimensions
        //float depth_width = depth.get_width();
        //float depth_height = depth.get_height();

        // Creating OpenCV Matrix from a color image
        
        Mat color(Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);

        // Display in a GUI
    	count++;
	if (count == 1||count==30)background = color.clone(); //提取第一帧为背景帧
	imshow("background",background);
        //imshow("frame",color);
	//imshow("video", frame);
	if (count>30)
	{
            result = MoveDetect(background, color,depth);
	    imshow("result", result);
	}
        
        if (waitKey(50) == 27)
            break;

        //imshow("Display Image", color);
        //bgSubtractor(color,mask,0.001);
	//imshow("mask",mask);

        //waitKey(0);
    }

    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
