#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Función para actualizar los valores de los límites HSV
void updateHsv(int, void*);

int main() {
    // Creamos una ventana para mostrar la imagen
    namedWindow("Image", WINDOW_NORMAL);

    // Creamos barras de desplazamiento para los valores HSV
    int hMin = 0, sMin = 0, vMin = 0;
    int hMax = 179, sMax = 255, vMax = 255;
    createTrackbar("Hue Min", "Image", &hMin, 179, updateHsv);
    createTrackbar("Sat Min", "Image", &sMin, 255, updateHsv);
    createTrackbar("Val Min", "Image", &vMin, 255, updateHsv);
    createTrackbar("Hue Max", "Image", &hMax, 179, updateHsv);
    createTrackbar("Sat Max", "Image", &sMax, 255, updateHsv);
    createTrackbar("Val Max", "Image", &vMax, 255, updateHsv);

    // Cargamos la imagen
    Mat image = imread("image.jpg");

    // Convertimos la imagen al espacio de color HSV
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);

    // Mostramos la imagen original
    imshow("Image", image);

    // Esperamos a que se presione una tecla
    waitKey(0);

    return 0;
}

void updateHsv(int, void*) {
    // Obtenemos los valores actuales de los límites HSV
    int hMin = getTrackbarPos("Hue Min", "Image");
    int sMin = getTrackbarPos("Sat Min", "Image");
    int vMin = getTrackbarPos("Val Min", "Image");
    int hMax = getTrackbarPos("Hue Max", "Image");
    int sMax = getTrackbarPos("Sat Max", "Image");
    int vMax = getTrackbarPos("Val Max", "Image");

    // Creamos un rango de valores HSV
    Scalar lower(hMin, sMin, vMin);
    Scalar upper(hMax, sMax, vMax);

    // Filtramos la imagen utilizando el rango de valores HSV
    Mat filteredImage;
    inRange(hsvImage, lower, upper, filteredImage);

    // Mostramos la imagen filtrada
    imshow("Image", filteredImage);
}