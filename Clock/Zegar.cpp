#include <opencv2/opencv.hpp>
#include <math.h>
#define IMAGES_PATH "Pictures"

//BLUR
//#define blurKernel cv::Size(11,11)
#define gaussKernel_pre cv::Size(19,19)
#define gaussKernel_seg cv::Size(5,5)
#define gaussSigma 0
//KONTRAST
#define startKontrast_pre 0
#define endKontrast_pre 100
#define startKontrast_seg 20
#define endKontrast_seg 115
//WYKRYWANIE OKRÊGÓW
#define wspKorekcjiPromienia 0.70 //Wspó³czynnik dla promienia ko³a przy wycinaniu ROI
//KONTUROWANIE
#define wspPromienKolaKonturowanie 0.25 //Stosunek promienia nak³adanego ko³a do promienia obrazu po preprocesingu
#define wspAR 5 //Minimalna wartoœæ wspó³czynnika Aspect Ratio
#define minPole 25 //Minimalne pole BBox aby kontur zosta³ wykryty
#define wspMaxPole 0.10 //Maksymalna wartoœæ obrazu zajmowana przez BBox konturu
//PROGOWANIE SEGMENTACJA
#define ProgowanieSegmentacja 127 //Próg dla finalnego progowania

#define DEBUG 0 //Debug Mode - pokazywanie dzia³ania programu krok po kroku

struct daneAnaliza
{
	int offX; //Offset obrazu po preprocesingu (ROI)
	int offY;
	int wskazowka1Dlugosc; //Dlugosci wskazowek
	int wskazowka2Dlugosc;
	double kat1; //Wyznaczone k¹ty
	double kat2;
	int Godzina; //Wyniki analizy - czas
	int Minuta;
};

inline void wait(int time = 0)
{
	if (cv::waitKey(time) == 'q') exit(0);
}

inline void debugshow(const cv::Mat img) 
{
#if DEBUG == 1
	cv::imshow("Debug", img);
	wait();
#endif
}

cv::Mat progowanie(const cv::Mat img, int prog, int type)
{
	cv::threshold(img, img, prog, 255, type);
	debugshow(img);
	return img;
}

cv::Mat kontrast(const cv::Mat img, int min, int max) //dla 8-bitowej g³êbi
{
	cv::Mat tmp = img.clone();
	long mid = (min + max) / 2;
	for (int x = 0; x < tmp.rows; x++)
	{
		for (int y = 0; y < tmp.cols; y++)
		{
			long pix = tmp.at<uchar>(x, y);
			if (pix > max)
			{
				tmp.at<uchar>(x, y) = 255;
			}
			else if (pix < min)
			{
				tmp.at<uchar>(x, y) = 0;
			}
			else 
			{
				tmp.at<uchar>(x, y) = (int)abs(127.5 + (double)((pix- mid)*255.0) / (double)(max+ min));
			}
		}
	}
	debugshow(tmp);
	return tmp;
}

void konturowanie(const cv::Mat img, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i>& hierarchy)
{
	std::vector<std::vector<cv::Point>> c_tmp;
	std::vector<cv::Vec4i> h_tmp;
	cv::findContours(img, c_tmp, h_tmp, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < c_tmp.size(); ++i)
	{
		if (h_tmp[i][3] < 0) //element nadrzêdny
		{
			cv::RotatedRect r_rect = cv::minAreaRect(c_tmp[i]);
			bool warunek1 = (r_rect.size.width / r_rect.size.height) > wspAR || r_rect.size.height / r_rect.size.width > wspAR;
			bool warunek2 = (r_rect.size.width * r_rect.size.height) > minPole;
			bool warunek3 = (r_rect.size.width * r_rect.size.height) < wspMaxPole * img.cols * img.rows;
			if (warunek1 && warunek2 && warunek3) //3 warunki wymagane do spe³nienia
			{
				contours.push_back(c_tmp[i]);
				hierarchy.push_back(h_tmp[i]);
			}
		}
	}
}

cv::Mat rysujKontury(const cv::Mat img, const std::vector<std::vector<cv::Point>>& contours, const std::vector<cv::Vec4i>& hierarchy)
{
	cv::Mat tmp(img.size(), CV_8UC3);
	tmp = cv::Scalar::all(255);
	for (int i = 0; i < contours.size(); ++i)
	{
		cv::drawContours(tmp, contours, i, cv::Scalar(0, 0, 255), 1);
	}
	debugshow(tmp);
	return tmp;
}

cv::Vec3f wykryjKola(cv::Mat img)
{
	std::vector<cv::Vec3f> wektorOkregi;
	int promien_max;
	cv::Vec3f okrag;
	cv::HoughCircles(img, wektorOkregi, cv::HOUGH_GRADIENT, 2, img.cols / 8, 100, 100, img.cols / 4);
	promien_max = 0;
	if (wektorOkregi.size() == 1) okrag = wektorOkregi[0]; //Je¿eli wykryto tylko 1 okr¹g
	else if (wektorOkregi.size() < 1) okrag = cv::Vec3f(0,0,0); //Je¿eli nie wykryto ¿adnego okrêgu
	else
	{
		for (size_t i = 0; i < wektorOkregi.size(); i++)
		{
			if (cvRound(wektorOkregi[i][2]) > promien_max)
			{
				okrag = wektorOkregi[i];
				promien_max = wektorOkregi[i][2];
			}
		}
	}
	okrag[2] = (int)(okrag[2] * wspKorekcjiPromienia); //Korekcja promienia okrêgu
	return okrag;
}

cv::Mat rysujKolo(cv::Mat img, cv::Vec3f wykryteKolo, cv::Scalar kolor)
{
	cv::Point srodek(cvRound(wykryteKolo[0]), cvRound(wykryteKolo[1]));
	int promien = cvRound(wykryteKolo[2]);
	circle(img, srodek, 3, kolor, -1, 8, 0);
	circle(img, srodek, promien, kolor, 1, 8, 0);
	debugshow(img);
	return img;
}

cv::Mat ROI_circles(cv::Mat img, cv::Vec3f wykryteKolo)
{
	cv::Mat tmp(img.size(), CV_8UC3);
	cv::Mat tmp2(img.size(), img.type());
	tmp = cv::Scalar::all(255);
	cv::Point srodek(cvRound(wykryteKolo[0]), cvRound(wykryteKolo[1]));
	int promien = cvRound(wykryteKolo[2]);
	// Rysowanie maski - bia³e ko³o na czarnym tle
	cv::Mat1b mask(img.size(), uchar(0));
	circle(mask, srodek, promien, cv::Scalar(255), cv::FILLED);
	// Obliczenie prostok¹ta "Bounding box" zamykaj¹cego okr¹g
	cv::Rect bbox(wykryteKolo[0] - promien, wykryteKolo[1] - promien, 2 * promien, 2 * promien);
	// Stworzenie bia³ego obrazu
	tmp2 = cv::Scalar::all(255);
	// Skopiowanie z wykorzystaniem maski
	img.copyTo(tmp2, mask);
	// Przyciêcie
	tmp2 = tmp2(bbox);
	debugshow(tmp2);
	return tmp2;
}

double obliczKat(cv::Vec<cv::Point2f, 2> punkty1, cv::Point srodek)
{
	int y[2], x[2];
	y[0] = srodek.y - punkty1[0].y;
	y[1] = srodek.y - punkty1[1].y;
	x[0] = punkty1[0].x - srodek.x;
	x[1] = punkty1[1].x - srodek.x;
	if (x[0]*x[0]+y[0]*y[0] > x[1]*x[1]+y[1]*y[1]) //ZAMIANA (zawsze x1,y1 jest blizej srodka)
	{
		int ytmp, xtmp;
		xtmp = x[0];
		ytmp = y[0];
		y[0] = y[1];
		x[0] = x[1];
		y[1] = ytmp;
		x[1] = xtmp;
	}
	double kat = 180.0 / 3.14 * atan(abs((double)(y[1] - y[0])) / abs((double)(x[1] - x[0])));
	//Transformuj zale¿nie od po³o¿enia w przyjêtym uk³adzie
	if (x[1] > 0 && y[1] > 0) kat = 90 - kat; //cwiartka I
	else if (x[1] < 0 && y[1] > 0) kat = kat + 270; //cwiartka II
	else if (x[1] < 0 && y[1] < 0) kat = 270 - kat; //cwiartka III
	else kat = kat + 90; //cwiarka IV
	return kat;
}

int obliczGodz(double kat)
{
	int godz = ((int)(floor(kat / 30.0)));
	if (godz == 0) godz = 12;
	return godz;
}

int obliczMin(double kat)
{
	return ((int)(floor(kat / 6.0)));
}

void zegar(daneAnaliza* dane)
{
	if (dane->wskazowka1Dlugosc > dane->wskazowka2Dlugosc) //Minuta, godzina
	{
		dane->Godzina = obliczGodz(dane->kat2);
		dane->Minuta = obliczMin(dane->kat1);
	}
	else //Godzina, minuta
	{
		dane->Godzina = obliczGodz(dane->kat1);
		dane->Minuta = obliczMin(dane->kat2);
	}
}

int preprocesing(cv::Mat* img, cv::Mat* img_o, daneAnaliza* dane)
{
	cv::Mat tmp = img->clone();
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::cvtColor(*img, tmp, cv::COLOR_BGR2GRAY);
	debugshow(tmp);
	//KONTRAST 
	tmp = kontrast(tmp, startKontrast_pre, endKontrast_pre);
	//BLUR
	cv::GaussianBlur(tmp, tmp, gaussKernel_pre, gaussSigma, 0);
	debugshow(tmp);
	//WYKRYJ KO£A
	cv::Vec3f wykryteKolo = wykryjKola(tmp);
	if (wykryteKolo == cv::Vec3f(0, 0, 0)) //Nie wykryto ko³a
	{
		std::cerr << "ERROR! Clock face not detected. Check the picture and/or analysis parameters." << std::endl;
		return -1;
	}
	tmp = rysujKolo(tmp, wykryteKolo, cv::Scalar(0, 0, 255)); //Rysuj ko³o na obrazie tymczasowym (DEBUG)
	*img_o = rysujKolo(*img_o, wykryteKolo, cv::Scalar(0,255,0)); //Rysuj ko³o na obrazie oryginalnym
	//ROI
	*img = ROI_circles(*img, wykryteKolo); //Wyciêcie w obrazie
	//ZAPISZ OFFSET
	dane->offX = wykryteKolo[0] - img->cols / 2;
	dane->offY = wykryteKolo[1] - img->rows / 2;
	return 1;
}

void segmentacja(cv::Mat* img, cv::Mat* img_o)
{
	cv::cvtColor(*img, *img, cv::COLOR_BGR2GRAY);
	debugshow(*img);
	//BLUR
	cv::GaussianBlur(*img, *img, gaussKernel_seg, gaussSigma, 0);
	debugshow(*img);
	//KONTRAST 
	*img = kontrast(*img, startKontrast_seg, endKontrast_seg);
	//BLUR
	cv::GaussianBlur(*img, *img, gaussKernel_seg, gaussSigma, 0);
	debugshow(*img);
	//PROGOWANIE
	*img = progowanie(*img, ProgowanieSegmentacja, cv::THRESH_BINARY);
	//DYLATACJA
	cv::dilate(*img, *img, cv::noArray());
	debugshow(*img);
	//EROZJA
	cv::erode(*img, *img, cv::noArray());
	debugshow(*img);
	//INWERSJA
	*img = ~*img;
}

int analiza(cv::Mat* img, cv::Mat* img_o, daneAnaliza* dane)
{
	cv::Point srodek = cv::Point(img->cols / 2, img->rows / 2);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Vec<cv::Point2f, 4> prostokatPunkty[2]; //Punkty zwi¹zane z prostok¹tami "bounding box"
	cv::Vec<cv::Point2f, 2> osPunkty[2]; //Punkty zwi¹zane z osiami wskazówek
	cv::RotatedRect prost[2];
	std::stringstream ss;
	//NA£O¯ENIE KO£A NA ŒRODEK TARCZY
	circle(*img, cv::Point(img->cols / 2, img->rows / 2), wspPromienKolaKonturowanie * img->rows / 2, cv::Scalar::all(0), cv::FILLED);
	//KONTUROWANIE
	konturowanie(*img, contours, hierarchy);
	*img = rysujKontury(*img, contours, hierarchy);
	circle(*img, srodek, 1, cv::Scalar(0, 255, 0), cv::FILLED);
	debugshow(*img);
	//SPRAWDZENIE LICZBY KONTURÓW
	if (contours.size() == 0)
	{
		std::cerr << "ERROR! Contours not detected. Check the picture and/or analysis parameters." << std::endl;
		return -1;
	}
	else if (contours.size() < 2) //wykryto tylko jedn¹ wskazówkê
	{
		contours.push_back(contours[0]); //powielenie konturu
		hierarchy.push_back(hierarchy[0]); //i jego hierarchii
	}
	//OPISANIE PROSTOK¥TEM
	prost[0] = cv::minAreaRect(contours[0]);
	prost[0].points(&prostokatPunkty[0][0]);
	prost[1] = cv::minAreaRect(contours[1]);
	prost[1].points(&prostokatPunkty[1][0]);
	//POBRANIE D£UGOŒCI I OSI
	if (prost[0].size.aspectRatio() > 1)
	{
		dane->wskazowka1Dlugosc = prost[0].size.width;
		osPunkty[0][0] = prostokatPunkty[0][1];
		osPunkty[0][1] = prostokatPunkty[0][2];
	}
	else
	{
		dane->wskazowka1Dlugosc = prost[0].size.height;
		osPunkty[0][0] = prostokatPunkty[0][0];
		osPunkty[0][1] = prostokatPunkty[0][1];
	}
	if (prost[1].size.aspectRatio() > 1)
	{
		dane->wskazowka2Dlugosc = prost[1].size.width;
		osPunkty[1][0] = prostokatPunkty[1][1];
		osPunkty[1][1] = prostokatPunkty[1][2];
	}
	else
	{
		dane->wskazowka2Dlugosc = prost[1].size.height;
		osPunkty[1][0] = prostokatPunkty[1][0];
		osPunkty[1][1] = prostokatPunkty[1][1];
	}
	//WYRYSOWANIE LINII OSI
	cv::line(*img, osPunkty[0][0], osPunkty[0][1], cv::Scalar(255, 0, 0));
	cv::line(*img, osPunkty[1][0], osPunkty[1][1], cv::Scalar(255, 0, 0));
	cv::line(*img_o, cv::Point(osPunkty[0][0].x+dane->offX, osPunkty[0][0].y + dane->offY), cv::Point(osPunkty[0][1].x + dane->offX, osPunkty[0][1].y + dane->offY), cv::Scalar(255, 0, 0),2);
	cv::line(*img_o, cv::Point(osPunkty[1][0].x + dane->offX, osPunkty[1][0].y + dane->offY), cv::Point(osPunkty[1][1].x + dane->offX, osPunkty[1][1].y + dane->offY), cv::Scalar(255, 0, 0),2);
	//WYZNACZ K¥T D£U¯SZEGO BOKU
	dane->kat1 = obliczKat(osPunkty[0], srodek);
	dane->kat2 = obliczKat(osPunkty[1], srodek);
	//ZAPISZ K¥T
	ss << dane->kat1;
	cv::putText(*img, ss.str(), cv::Point(prostokatPunkty[0][0]), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar::all(0));
	cv::putText(*img_o, ss.str(), cv::Point(prostokatPunkty[0][0].x+dane->offX, prostokatPunkty[0][0].y + dane->offY), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar::all(0));
	ss.str("");
	ss << dane->kat2;
	cv::putText(*img, ss.str(), cv::Point(prostokatPunkty[1][0]), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar::all(0));
	cv::putText(*img_o, ss.str(), cv::Point(prostokatPunkty[1][0].x + dane->offX, prostokatPunkty[1][0].y + dane->offY), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar::all(0));
	//WYZNACZ GODZINE
	zegar(dane);
	//ZAPISZ GODZINE NA OBRAZACH
	std::cout << dane->Godzina << ":" << dane->Minuta << std::endl;
	ss.str("");
	ss << std::setw(2) << std::setfill('0') << dane->Godzina << ":";
	ss << std::setw(2) << std::setfill('0') << dane->Minuta;
	cv::putText(*img, ss.str(), cv::Point(img->cols/2,img->rows-5), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar::all(0));
	cv::putText(*img_o, ss.str(), cv::Point(img->cols/2+dane->offX, img->rows / 2 + dane->offY), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));
	ss.str("Kacper-Marciniak");
	cv::putText(*img_o, ss.str(), cv::Point(1, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
	return 1;
}

void zapiszObrazKoncowy(const cv::Mat img, int i)
{
	std::stringstream ss;
	ss.str("");
	ss << IMAGES_PATH << "\\out_" << i << ".jpg";
	cv::imwrite(ss.str(), img);
}

int main()
{
	for (int i = 1; 1; ++i)
	{
		std::stringstream ss;
		cv::Mat img_o, imgTmp;
		cv::Mat* wskImg_o = &img_o;
		cv::Mat* wskImgTmp = &imgTmp;
		daneAnaliza daneA;
		daneAnaliza *wskDaneA = &daneA;
		//WCZYTANIE
		std::cout << "Analysis nr.: " << i << std::endl;
		ss << IMAGES_PATH << "\\"  << i << ".jpg";
		*wskImg_o = cv::imread(ss.str());
		if (img_o.rows < 1 || wskImg_o->cols < 1) break; //WyjdŸ z pêtli je¿eli obraz wczytany jest pusty
		cv::imshow("Oryginal picture", img_o);
		*wskImgTmp = wskImg_o->clone();
		//PRZETWARZANIE I ANALIZA
		preprocesing(wskImgTmp, wskImg_o, wskDaneA);
		cv::imshow("After preprocessing", *wskImgTmp);
		segmentacja(wskImgTmp, wskImg_o);
		cv::imshow("After segmentation", *wskImgTmp);
		if (analiza(wskImgTmp, wskImg_o, wskDaneA) == -1) { wait(); continue; }
		cv::imshow("After analysis", *wskImgTmp);
		cv::imshow("Oryginal picture after analysis", img_o);
		wskImgTmp->convertTo(*wskImgTmp, wskImg_o->type());
		//ZAPIS DO PLIKU
		zapiszObrazKoncowy(*wskImg_o, i);
		wait();
		cv::destroyAllWindows();
	}
	std::cout << "Program finished its work" << std::endl;
	wait();
	cv::destroyAllWindows();
	return 1;
}