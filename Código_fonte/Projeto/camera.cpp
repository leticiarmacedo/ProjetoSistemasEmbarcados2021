#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <unistd.h>
using namespace std;
using namespace cv;

const int fps = 30;

int main(int argc, char** argv){
	Mat frame;
	VideoCapture vid(0);
	if(!vid.isOpened()){
		printf("Não foi possóvel encontrar a câmera!");
		return -1;
	}
	while(1){
		vid.read(frame);
		namedWindow("webcam", WINDOW_AUTOSIZE);
		imshow("webcam",frame);
		waitKey(1000/fps);
	}
	return 0;
}
