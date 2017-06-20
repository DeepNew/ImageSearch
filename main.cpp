//%***********************************************%
//% This code comes from the matlab code of       % 
//% Liang Zheng                                   %
//% Please modify the path to your own folder.    %
//% Author by: Ren Yimo                           %
//%***********************************************%
//% if you find this code useful in your research, please kindly cite the
//% paper as,
//% Liang Zheng, Liyue Sheng, Lu Tian, Shengjin Wang, Jingdong Wang, and Qi Tian,
//% Scalable Person Re-identification: A Benchmark, ICCV, 2015.

#include <iostream>
#include "opencv2\contrib\contrib.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include<time.h>
//windows
#include <io.h>
#include <direct.h>  
//linux
//#include <unistd.h>  
//#include <sys/types.h>  
//#include <sys/stat.h>  
using namespace std;
using namespace cv;
Mat readImg(string imagename)
{
	//从文件中读入图像
    Mat img = imread(imagename);
    //如果读入图像失败
    if(img.empty())
    {
        fprintf(stderr, "Can not load image %s\n", imagename);
    }
	return img;
}
Mat getCodebook(int clusterNum)
{
	Mat codebook;
	if (clusterNum == 350)
	{     
		string outpath = "codebook_350.xml";
		FileStorage fs(outpath,FileStorage::READ);  
		fs["codebook_350"]>>codebook;  
		return codebook;
	}
	if (clusterNum == 500)
	{
		//only support when the size of codebook is 500 
		string outpath = "model\\codebook_500.xml";
		FileStorage fs(outpath,FileStorage::READ);  
		fs["codebook_500"]>>codebook;  
		return codebook;
	}

}
Mat getNE()
{
	string outpath = "model\\ne_500.xml";
	FileStorage fs(outpath,FileStorage::READ);
	Mat ne;
	fs["ne_500"]>>ne;  
	return ne;
}
Mat getIDF()
{
	string outpath = "model\\idf_500.xml";
	FileStorage fs(outpath,FileStorage::READ);
	Mat idf;
	fs["idf_500"]>>idf;  
	return idf;
}
Mat getW2C()
{
	string outpath = "model\\w2c.xml";
	FileStorage fs(outpath,FileStorage::READ);
	Mat w2c;
	fs["w2c"]>>w2c;  
	return w2c;
}
Mat floorMat(const Mat & doubleMat)  
{  
    int rows = doubleMat.rows;  
    int cols = doubleMat.cols;  
    Mat flo(rows,cols,CV_32SC1);  
  
    for(int r = 0;r < rows;r++)  
    {  
        for(int c = 0;c < cols ;c++)  
        {  
            flo.at<int>(r,c)= floor(doubleMat.at<double>(r,c)/8.0);  
        }  
    }  
    return flo;  
}  
Mat im2c(Mat image,Mat w2c,int color)
{
	//input im should be DOUBLE !
	//color=0 is color names out
	//color=-1 is colored image with color names out
	//color=1-11 is prob of colorname=color out;
	//color=-1 return probabilities
	//order of color names: black ,   blue   , brown       , grey       , green   , orange   , pink     , purple  , red     , white    , yello
	//color_values =     {  [0 0 0] , [0 0 1] , [.5 .4 .25] , [.5 .5 .5] , [0 1 0] , [1 .8 0] , [1 .5 1] , [1 0 1] , [1 0 0] , [1 1 1 ] , [ 1 1 0 ] };
	/* ----------------------11种颜色值----------------------------------*/  
	Mat color_values(11,1,CV_64FC3);  
  
    //black-黑色 [0 0 0]  
    color_values.at<Vec3d>(0,0) = Vec3d(0,0,0);  
  
    //blue-蓝色 [0 0 1]  
    color_values.at<Vec3d>(1,0) = Vec3d(0,0,1);  
  
    //brown-棕色(褐色) [0.5 0.4 0.25]  
    color_values.at<Vec3d>(2,0) = Vec3d(0.5,0.4,0.25);  
  
    //grey-灰色[0.5 0.5 0.5]  
    color_values.at<Vec3d>(3,0) = Vec3d(0.5,0.5,0.5);  
  
    //green-绿色[0 1 0]  
    color_values.at<Vec3d>(4,0) = Vec3d(0,1,0);  
  
    //orange-橘色[1 0.8 0]  
    color_values.at<Vec3d>(5,0) = Vec3d(1,0.8,0);  
  
    //pink-粉红色[1 0.5 1]  
    color_values.at<Vec3d>(6,0) = Vec3d(1,0.5,1);  
  
    //purple-紫色[1 0 1]  
    color_values.at<Vec3d>(7,0) = Vec3d(1,0,1);  
  
    //red-红色 [1 0 0]  
    color_values.at<Vec3d>(8,0) = Vec3d(1,0,0);  
  
    //white-白色 [1 1 1]  
    color_values.at<Vec3d>(9,0) = Vec3d(1,1,1);  
  
    //yellow-黄色[1 1 0]  
    color_values.at<Vec3d>(10,0) = Vec3d(1,1,0);
	/* ----------------------------------------------------------------*/  
    int rows = image.rows;  
    int cols = image.cols;  
    int areas = rows*cols;  
      
    //分离通道  
    vector<Mat> bgr_planes;  
    split(image,bgr_planes);  
  
    //把各通道转为64F  
    Mat bplanes,gplanes,rplanes;  
    bgr_planes[0].convertTo(bplanes,CV_64FC1);  
    bgr_planes[1].convertTo(gplanes,CV_64FC1);  
    bgr_planes[2].convertTo(rplanes,CV_64FC1); 

	//floor(各通道/8.0)     
	Mat fbplanes,fgplanes,frplanes;  
    fbplanes = floorMat(bplanes);  
    fgplanes = floorMat(gplanes);  
    frplanes = floorMat(rplanes);  
	Mat index_im = frplanes+32*fgplanes+32*32*fbplanes;//index_im最大值可能为:31+31*32+32*32*31=32767  
	index_im=index_im.reshape(0,areas);
	
	if (color == -2)
	{
		//Mat out(Size(w2c.cols,index_im.rows), CV_64FC1, Scalar(0));
		Mat out(index_im.rows, w2c.cols,  CV_32FC1);  

		for (int i = 0;i<index_im.rows;i++)
		{ 
			w2c.row(index_im.at<int>(i)).copyTo(out.row(i));
		}
		//求每一列的均值
		Mat out2(1, w2c.cols,  CV_32FC1);
		for(int j=0;j<out.cols;j++)
		{
			Mat submat = out.col(j);
			Scalar tempVal = mean(submat);  
			float matMean = tempVal.val[0];  
			out2.at<float>(0,j)=matMean;
		}
		return out2;
	}
	
}
void meshgrid(int xgv[3], int ygv[3], cv::Mat &X, cv::Mat &Y)  
{  
    std::vector<float> t_x, t_y;  
    for(int i = 0; i < xgv[2]-1; i++) 
	{
		t_x.push_back(float(xgv[0]+i*(float(xgv[1])-float(xgv[0]))/float(xgv[2])));
	}
	t_x.push_back(xgv[1]);

    for(int i = 0; i < ygv[2]-1; i++) 
	{
		t_y.push_back(float(ygv[0]+i*(float(ygv[1])-float(ygv[0]))/float(ygv[2])));
	}
	t_y.push_back(ygv[1]);

    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);  
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);  
}  
Mat getCNDescriptor(Mat img,Mat w2c,Mat codebook)
{
	// Given an input image, this code calculates a 8000-dim feature vector 
	int maxK_knn = 10;
	int step = 4;
	int cellW = 4;
	int cellH = 4;
	int imgH = img.rows;
	int imgW = img.cols;
	const int cellX = (imgW-cellW)/step+1;
	const int cellY = (imgH-cellH)/step+1;
	int nwords= codebook.rows;

	int sz[] = {maxK_knn,cellY,cellX};  
	Mat Words = Mat(3, sz, CV_32FC1, Scalar::all(0));  //words name
	Mat Dwords = Mat(3, sz, CV_32FC1, Scalar::all(0));  //distance of words

	//for each patch, extract CN descriptor, followed by quantization with MA = 10
	Mat idf=getIDF();
	Mat ne=getNE();
	
	int fLength = 11;//nwords?
	int sz1[] = {fLength,cellY,cellX}; 
	Mat Feature = Mat(3, sz1, CV_32FC1, Scalar::all(0)); 
	
	//img=double(img)
	for (int j = 0;j<cellY;j++)
	{
		for(int i = 0;i<cellX;i++)
		{
			//extract one cell size(cellH,cellW)
			Mat data = img(Range(0+j*step,cellH+j*step),Range(0+i*step,cellW+i*step)).clone();
			Mat tempbin = im2c(data,w2c,-2); //feature bin
			Mat tempnorm(nwords, 1,  CV_32FC1,Scalar(0));//distance between nth words and the testing image
			for (int k = 0;k<nwords;k++)
			{
				tempnorm.at<float>(k,0)=norm(tempbin,codebook.row(k),NORM_L2);				
			}
			//c1 is sorted mat ，c2 is index
			Mat c1(tempnorm.rows,tempnorm.cols, CV_32FC1);
			tempnorm.copyTo(c1);
			Mat c2;
			sortIdx(c1,c2, SORT_EVERY_COLUMN + SORT_ASCENDING);
			cv::sort(c1,c1,SORT_EVERY_COLUMN + SORT_ASCENDING);

			//Feature(:,j,i) = tempbin;
			for (int k =0;k<tempbin.cols;k++)
				Feature.at<float>(k,j,i)=tempbin.at<float>(0,k);
			//Words(:,j,i) = order(1:maxK_knn);
			
			for (int k = 0;k<maxK_knn;k++)
			{
				Words.at<float>(k,j,i)=float(c2.at<int>(k,0));
			}
			//Dwords(:,j,i) = D(1:maxK_knn);
			for (int k = 0;k<maxK_knn;k++)
				Dwords.at<float>(k,j,i)=c1.at<float>(k,0);
		}
	}
	// Gaussian mask for background suppression
	int flag_bu = 1;
	int flag_gauss = 1;
	int ystep = 2;
	int striplength = 2;
	int nstrip=(cellY-striplength)/ystep+1;
	int k_knn = 10;
	int sigma = 3;

	// Wwords = exp(-Dwords/(sigma^2));
	Mat Wwords = Mat(3, sz, CV_32FC1, Scalar::all(0));
	for (int i = 0;i<sz[0];i++)
		for (int j = 0;j<sz[1];j++)
			for(int k =0;k<sz[2];k++)
				Wwords.at<float>(i,j,k)=exp(-(Dwords.at<float>(i,j,k)/(sigma*sigma)));
	if (flag_gauss == 1)
	{
		Mat w_g_m = Mat(3, sz, CV_32FC1, Scalar::all(0));
		//[x,y] = meshgrid(linspace(-1,1,cellX),linspace(-1,1,cellY));
		Mat x,y;
		int xgv[3]={-1,1,cellX};
		int ygv[3]={-1,1,cellY};
		meshgrid(xgv,ygv,x,y);

		int sigmax = 1;
		int sigmay = 1;
		//w_g = exp(-(x/sigmax).^2-(y/sigmay).^2);
		Mat w_g(x.rows,x.cols,CV_32FC1);
		for (int i =0;i<x.rows;i++)
			for(int j = 0;j<x.cols;j++)
				w_g.at<float>(i,j)=exp(-((x.at<float>(i,j)/sigmax)*(x.at<float>(i,j)/sigmax)+(y.at<float>(i,j)/sigmay)*(y.at<float>(i,j)/sigmay)));
		
		for (int k =0;k<maxK_knn;k++)
			for (int i = 0;i<cellX;i++)
				for(int j = 0;j<cellY;j++)
					w_g_m.at<float>(k,j,i) = w_g.at<float>(j,i);
		for (int k =0;k<maxK_knn;k++)
			for (int i = 0;i<cellX;i++)
				for(int j = 0;j<cellY;j++)
					Wwords.at<float>(k,j,i)*=w_g_m.at<float>(k,j,i);	
	}
	//calculate final feature vector for an image
	if (k_knn<10)
		for (int k =k_knn;k<maxK_knn;k++)
			for (int i = 0;i<cellX;i++)
				for(int j = 0;j<cellY;j++)
					Wwords.at<float>(k,j,i)=0;	
	Mat temphist(nwords,nstrip,CV_32FC1, Scalar::all(0));
	Mat histword(cellY,nwords,CV_32FC1, Scalar::all(0));
	int index = 0;
	for(int j = 0;j<cellY;j++)
	{
		for(int k = 0;k<cellX;k++)
		{
			index+=1;
			Mat tempWords(k_knn,1,CV_32FC1, Scalar::all(0));
			Mat tempWwords(k_knn,1,CV_32FC1, Scalar::all(0));

			for(int i =0;i<k_knn;i++)
			{
				tempWords.at<float>(i,0)=Words.at<float>(i,j,k);
				tempWwords.at<float>(i,0)=Wwords.at<float>(i,j,k);
			}
			tempWwords=tempWwords.reshape(0,1);
			
			for (int i = 0;i<k_knn;i++)
			{
				float temp=histword.at<float>(j,tempWords.at<float>(i,0));
				temp +=tempWwords.at<float>(0,i);
				histword.at<float>(j,tempWords.at<float>(i,0))=temp;
			}
		}
	}
	for(int j = 0;j<nstrip-1;j++)
	{
		Mat temphistword=histword(Range(j*ystep,j*ystep+striplength),Range(0,nwords));
		for (int i=0;i<nwords;i++)
		{
			temphist.at<float>(i,j)=temphistword.at<float>(0,i)+temphistword.at<float>(1,i);
			//cout<<temphist.at<float>(i,j)<<endl;deviation
		}
	}
	
	Mat temphistword=histword(Range((nstrip-2)*ystep+striplength,cellY),Range(0,nwords));
	
	for (int i = 0; i<nwords;i++)
	{
		float sum=0;
		
		for(int j = 0;j<cellY-((nstrip-2)*ystep+striplength);j++)
		{
			sum+=temphistword.at<float>(j,i);
		}
		temphist.at<float>(i,nstrip-1)=sum;
	}
	int sz2[] = {k_knn,cellY,cellX};  
	Mat a = Mat(3, sz2, CV_32FC1, Scalar::all(0));  //words name
	for(int i =0;i<sz2[0];i++)
		for(int j = 0;j<sz2[1];j++)
			for(int k =0;k<sz2[2];k++)
				a.at<float>(i,j,k)=Words.at<float>(i,j,k);
	Mat b = Mat(1,sz2[0]*sz2[1]*sz2[2],CV_32FC1, Scalar::all(0));
	int count=0;
	for(int k =0;k<sz2[2];k++)
		for(int j =0;j<sz2[1];j++)
			for(int i=0;i<sz2[0];i++)
			{
				b.at<float>(count)=a.at<float>(i,j,k);
				count++;
			}
	
	Mat c;
	// bins :
	int histSize = nwords;
	// value range 
	float range[] = { 0, nwords } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	calcHist( &b, 1, 0, Mat(), c, 1, &histSize,&histRange, uniform, accumulate);
	transpose(c,c);

	Mat d(nstrip,1,CV_32FC1, Scalar::all(1));
	Mat e=d*c;
	
	Mat TF=e.reshape(0,e.rows*e.cols);
	for(int i =0;i<TF.rows;i++)
	{
		//cout<<TF.at<float>(i,0)<<endl;
		if(TF.at<float>(i,0)==0)
			TF.at<float>(i,0)+=1;
	}

	Mat Hist=temphist.reshape(0,temphist.rows*temphist.cols);
	//Mat Hist(1,temphist.rows*temphist.cols)
	if(flag_bu == 1)
	{
		//TF.rows==Hist.rows
		for(int i=0;i<Hist.rows;i++)
		{
			Hist.at<float>(i,0)/=sqrt(TF.at<float>(i,0));
		}
	}

	Mat out_hist(Hist.rows,Hist.cols, CV_32FC1, Scalar::all(0));
	for(int i =0;i<out_hist.rows;i++)
		for(int j =0;j<out_hist.cols;j++)
		{
			out_hist.at<float>(i,j)=(Hist.at<float>(i,j)-ne.at<float>(i,j))*sqrt(idf.at<float>(i,j));
		}
	return out_hist ;
}
Mat getCNDescriptorFromDir(string img_dir,Mat w2c,Mat codebook)
{
	Directory dir;  
	vector<string> fileNames = dir.GetListFiles(img_dir, "*.jpg", false);  
	int numPictures = fileNames.size(); //num of pictures

	vector<Mat>tempfeatures;
	for(int i =0;i<numPictures;i++)
	{
		string filename = img_dir + fileNames[i];	
		Mat img = readImg(filename);
		Mat feature = getCNDescriptor(img,w2c,codebook);
		tempfeatures.push_back(feature);
	}
	//normalization
	int Dim = tempfeatures[0].rows;
	int Num = tempfeatures.size();

	Mat features(Dim,Num,CV_32FC1, Scalar::all(0));
	for (int j=0; j<Num; j++)
    {
        (tempfeatures[j]).copyTo(features.col(j));
    }

	for (int i=0;i<Dim;i++)
		for (int j=0;j<Num;j++)
			features.at<float>(i,j)=features.at<float>(i,j)*features.at<float>(i,j);
	Mat temp_sum_val(1,Num,CV_32FC1, Scalar::all(0));
	for (int j = 0;j<Num;j++)
	{
		float sum=0;
		for(int i =0;i<Dim;i++)
		{
			sum+=features.at<float>(i,j);
		}
		sum=sqrt(sum);
		temp_sum_val.at<float>(0,j)=sum;
	}
	Mat sum_val(Dim,Num,CV_32FC1, Scalar::all(0));
	repeat(temp_sum_val, Dim, 1, sum_val);

	divide(features,sum_val,features,1,-1);
	//for(int i =0;i<features.rows;i++)
	//	cout<<features.at<float>(i,0)<<endl;
	return features;
}

Mat spdist(Mat Hist_gallary,Mat Hist_querry)
{
	int nQuerry = Hist_querry.cols;
	int nGallary = Hist_gallary.cols;
	
	Mat dist(nGallary,nQuerry,CV_32FC1, Scalar::all(0));
	for(int j = 0;j<nQuerry;j++)
	{
		for(int i = 0;i<nGallary;i++)
		{
			dist.at<float>(i,j)=norm(Hist_querry.col(j),Hist_gallary.col(i));
		}
	}
	return dist;
}

Mat re_id(string querrydir,string gallarydir)
{
	// clusterNum mean the number of words
	int clusterNum =500;
	// get codebokk
	Mat codebook = getCodebook(clusterNum);
	// used in CN extraction
	Mat w2c = getW2C();
	// calculate querry CN feature
	//string querrydir = "D:\\querry\\";
	Mat Hist_querry = getCNDescriptorFromDir(querrydir,w2c,codebook);
	int nQuerry = Hist_querry.cols;

	// calculate gallary CN feature
	//string gallarydir = "D:\\gallary\\";
	Mat Hist_gallary = getCNDescriptorFromDir(gallarydir,w2c,codebook);
	int nGallary = Hist_gallary.cols;

	//distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
	Mat dist = spdist(Hist_gallary,Hist_querry);

	//return index of result
	Mat result_index(nQuerry,nGallary,CV_32FC1,Scalar::all(0));
	for(int k =0;k<nQuerry;k++)
	{
		Mat score = dist.col(k);
		//c1 is sorted result ，c2 is index
		Mat c1(score.rows,score.cols,CV_32FC1,Scalar::all(0));
		score.copyTo(c1);
		Mat c2;
		sortIdx(c1,c2, SORT_EVERY_COLUMN + SORT_ASCENDING);
		cv::sort(c1,c1,SORT_EVERY_COLUMN + SORT_ASCENDING);
		transpose(c2,c2);
		c2.copyTo(result_index.row(k));
	}
	return result_index;
}
void int2str(const int &int_temp,string &string_temp)  
{  
        stringstream stream;  
        stream<<int_temp;  
        string_temp=stream.str(); 
}  
int main()
{  
	clock_t start_time=clock();
	string querrydir = "querry\\";
	string gallarydir = "gallary\\";

	Mat result_index=re_id(querrydir,gallarydir);
	Directory dir;  
	vector<string> qfileNames = dir.GetListFiles(querrydir, "*.jpg", false); 
	vector<string> fileNames = dir.GetListFiles(gallarydir, "*.jpg", false);  

	int nQuerry=result_index.rows;
	int nGallary=result_index.cols;
	clock_t end_time=clock();
	cout<< "Image Search Running time is: "<<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC<<"s"<<endl;
	cout<< "Querry Images Num is: "<<nQuerry<<endl;
	cout<< "Gallary Images Num is: "<<nGallary<<endl;
	//save the result 
	string resultdir = "result\\";
	for (int i = 0;i<nQuerry;i++)
	{
		int len = qfileNames[i].length();
		string outdir=resultdir+qfileNames[i].substr(0,len-4)+"\\";
		/*const char *path = outdir.c_str();
		int a=access(path,00);
		if(access(path,00)!=1)
		{
			if(_mkdir(path)==0){
				return 0;
			}
		}*/
		for(int j=0;j<nGallary;j++)
		{
			int index=int(result_index.at<float>(i,j));
			string filename = gallarydir+fileNames[index];
			Mat img = readImg(filename);
			string rank;
			int2str(j,rank);
			imwrite(outdir+rank+"_"+fileNames[index],img);
			//imshow("123",img);
			//waitKey(1000);
		}
	}
	
	system("PAUSE");
	return 0;
}
    
