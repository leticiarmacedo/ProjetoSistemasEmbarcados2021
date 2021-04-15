#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <time.h>
#include <signal.h>


using namespace dlib;
using namespace std;
using namespace cv;


image_window win;
shape_predictor sp;
std::vector<cv::Point> righteye;
std::vector<cv::Point> lefteye;
char c;
cv::Point p;
int const fps = 30;

double compute_EAR(std::vector<cv::Point> vec) // Razão de aspecto 
{

    double a = cv::norm(cv::Mat(vec[1]), cv::Mat(vec[5]));
    double b = cv::norm(cv::Mat(vec[2]), cv::Mat(vec[4]));
    double c = cv::norm(cv::Mat(vec[0]), cv::Mat(vec[3]));
    //compute EAR
    double ear = (a + b) / (2.0 * c);
    return ear;
}
my_handler(sig_t s){
           printf("Caught signal %d\n",s);
           exit(1); 

}
int main()
{
    try {
        cv::VideoCapture cap(0);

        if (!cap.isOpened()) {
            cerr << "Não foi possível encontrar a câmera" << endl;
            return 1;
        }
	//Definindo a Resolução
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

        //Definindo saída
	system("echo 4       > /sys/class/gpio/export");
	system("echo out      > /sys/class/gpio/gpio4/direction");
	system("echo 0      > /sys/class/gpio/gpio4/value");
	// Define o Detector de face e monta o modelo de aspectos do rosto.
        frontal_face_detector detector = get_frontal_face_detector();

        deserialize("../../landmarks/shape_predictor_68_face_landmarks.dat") >> sp;

        time_t inicio,momento;
        inicio = time(NULL);
        int last = 0;

        // Enquanto a janela estiver aberta processa o frame
        while (!win.is_closed()) {
            cv::Mat temp;			// Cria um frame
            if (!cap.read(temp)) {              // Se o programa não conseguir ler o frame sai do laço While.
                break;
            }

            cv_image<bgr_pixel> cimg(temp); 	//Converte Imagem
            full_object_detection shape;
            
            // Detecta faces
            std::vector<rectangle> faces = detector(cimg);
            cout << "Quantidade de rostos na imagem: " << faces.size() << endl;

            win.clear_overlay();		//Atualiza a tela
            win.set_image(cimg);
            
            // Encontra posição do rossto
            if (faces.size() > 0) {

                shape = sp(cimg, faces[0]); //Seleciona apenas o primeiro rosto

                for (int b = 36; b < 42; ++b) {
                    p.x = shape.part(b).x();
                    p.y = shape.part(b).y();
                    lefteye.push_back(p);
                }
                for (int b = 42; b < 48; ++b) {
                    p.x = shape.part(b).x();
                    p.y = shape.part(b).y();
                    righteye.push_back(p);
                }
                //Calcula a razão de aspecto dos olhos
                double right_ear = compute_EAR(righteye);
                double left_ear = compute_EAR(lefteye);

                //Se o resultado da razão de aspecto dos olhos for menor que 0.2 a pessoa está dormindo.
                if ((right_ear + left_ear) / 2 < 0.2){
                    if (last)
                    {
                        momento = time(NULL);
                        if(difftime(momento,inicio) >= 2){
                            win.add_overlay(dlib::image_window::overlay_rect(faces[0], rgb_pixel(255, 255, 255), "Dormindo"));
                            system("echo 1      > /sys/class/gpio/gpio4/value");
                        }
                    }
                    else{
                        inicio = time(NULL);
                    }
                    last = 1;
                }else{
                    last = 0;
                   	win.add_overlay(dlib::image_window::overlay_rect(faces[0], rgb_pixel(255, 255, 255), "Acordado"));
			system("echo 0      > /sys/class/gpio/gpio4/value");
		}
                righteye.clear();
                lefteye.clear();
                win.add_overlay(render_face_detections(shape));		//Adiciona os traços à imagem

                c = (char)waitKey(1000/fps);
                if (c == 27)
                    break;
            }
        }
	system("echo 4 > /sys/class/gpio/unexport");
    }
    catch (serialization_error& e) {
        cout << "Houve um erro na compilação do modelo de características faciais." << endl;
        cout << "Você consegue o arquivo completo através da  URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl
             << e.what() << endl;
    }
    catch (exception& e) {
        cout << e.what() << endl;
    }
	signal (SIGINT,my_handler);
}
