// ISODATA_Cluster.cpp : �������̨Ӧ�ó������ڵ㡣
//�Ե㲢��

#include "stdafx.h"
#include<vector>
#include<algorithm>
#include<set>
#include <stdlib.h> 
#include <omp.h>
#include<time.h>
#include<cstdlib>
#include<iostream>
#include <iterator> 
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 

using namespace std;  //ʹ��cout�����ʽ
using namespace cv;   // ʡȥ����ǰ���cv::�ı�Ҫ��


class isodata
{
private:
	unsigned int K;// ����Ҫ�ֳɵ������
	unsigned int thetaN;//һ���������Ӧ���е�������Ŀ����С�ڴ����Ͳ���Ϊһ�������ľ���
	double theta_c;// ��������֮��������Сֵ,���鲢ϵ������С�ڴ���������������кϲ�
	double theta_s;// һ�������������׼�����ֵ
	unsigned int maxcombine;// ÿ�ε������ɹ鲢����
	unsigned int maxiteration;// ����������
	unsigned int dim;   //����ά��
	double meandis;
	double alpha;
	Mat *Ori_Img;  //Դͼ��������
	Mat output;
	int n_Sample;
	unsigned int current_iter;
	double t_assign;
	double t_update_cen;
	double t_cacul_mean;
	double t_cacul_std;
	double t_split;
	double t_merge;
	int n_threads;
	vector<vector<int>>dataset;
	typedef vector<double> Centroid;
	struct position{
		int rows;
		int cols;
	};
	struct Cluster
	{
		Centroid center;       //��������
		vector<int>clusterID;   //���Ա���洢ID
		vector<position>pos;
		double inner_meandis;   //���ھ����ֵ
		vector<double>sigma;    //���ھ����׼��
	};
	vector<Cluster>clus;
private:
	void init();
	void assign();
	void check_thetaN();
	void update_centers();
	void update_center(Cluster &aa);
	void update_sigma(Cluster &aa);
	void calmeandis();
	void choose_nextstep();
	double distance(const Centroid &er, const int k);
	double distance(const Centroid &cen1, const Centroid &cen2);
	void split(const int kk);
	void check_for_split();
	void merge(const int k1, const int k2);
	void check_for_merge();
	void prepare_for_next_itration();
	void show_result();
	void  Id2RC(int &r, int &c, int id);

public:
	isodata()
	{
		time_t t;
		srand(time(&t));
	}
	void generate_data();
	void apply();
	void set_paras();
	void TMresize();
};
 
//���������չʾ
void isodata::show_result()
{
	Mat result = imread("p_B1.tif",IMREAD_COLOR);
	int cls_color[20][3] = { 0 };
	cls_color[0][0] = 255; cls_color[0][1] = 0; cls_color[0][2] = 255;  //��ɫ
	cls_color[1][0] = 255; cls_color[1][1] = 255; cls_color[1][2] = 0;  //��ɫ
	cls_color[2][0] = 128; cls_color[2][1] = 0; cls_color[2][2] = 128;  //����ɫ
	cls_color[3][0] = 0; cls_color[3][1] = 255; cls_color[3][2] = 255;  //��ɫ
	cls_color[4][0] = 255; cls_color[4][1] = 165; cls_color[4][2] = 0;  //��ɫ
	cls_color[5][0] = 0; cls_color[5][1] = 128; cls_color[5][2] = 0;  //����ɫ
	cls_color[6][0] = 0; cls_color[6][1] = 255; cls_color[6][2] = 0;  //��ɫ
	cls_color[7][0] = 0; cls_color[7][1] = 0; cls_color[7][2] = 255;  //��ɫ
	cls_color[8][0] = 230; cls_color[8][1] = 230; cls_color[8][2] = 250;  //����ɫ
	cls_color[9][0] = 210; cls_color[9][1] = 105; cls_color[9][2] = 30;  //��ɫ
	cls_color[10][0] = 128; cls_color[10][1] = 128; cls_color[10][2] = 0;  //���ɫ
	cls_color[11][0] = 0; cls_color[11][1] = 191; cls_color[11][2] = 255;  //������
	cls_color[12][0] = 255; cls_color[12][1] = 127; cls_color[12][2] = 80;  //ɺ��
	cls_color[13][0] = 240; cls_color[13][1] = 230; cls_color[13][2] = 140;  //���䲼
	cls_color[14][0] = 255; cls_color[14][1] = 192; cls_color[14][2] = 203;  //�ۺ�
	cls_color[15][0] = 255; cls_color[15][1] = 127; cls_color[15][2] = 80;  //ɺ��
	cls_color[16][0] = 240; cls_color[16][1] = 230; cls_color[16][2] = 140;  //���䲼
	cls_color[17][0] = 255; cls_color[17][1] = 192; cls_color[17][2] = 203;  //�ۺ�
	int r = 0, c = 0;
	for (int i = 0; i < clus.size(); i++)
	{
		for (int j = 0; j < clus[i].clusterID.size(); j++)
		{
			Id2RC(r,c,clus[i].clusterID[j]);
			result.at<Vec3b>(r,c)[0] = cls_color[i][2];
			result.at<Vec3b>(r,c)[1] = cls_color[i][1];
			result.at<Vec3b>(r,c)[2] = cls_color[i][0];
		}
	}
	imwrite("result.bmp",result);
	cout<<endl;
	cout<<"������ʱ�䣺        "<<t_assign<<endl;
	cout<<"���¾���������ʱ��: "<<t_update_cen<<endl;
	cout<<"����ƽ��������ʱ�䣺"<<t_cacul_mean<<endl;
	cout<<"�������׼����ʱ�䣺"<<t_cacul_std<<endl;
	//cout<<"������ʱ�䣺        "<<t_split<<endl;
	//cout<<"�ϲ���ʱ�䣺        "<<t_merge<<endl;
}
 
//��������ȡӰ������
void isodata::generate_data()
{
	dim = 6;
	Ori_Img = new Mat[6];
	string Img_Path[6] = {"p_B1.tif","p_B2.tif","p_B3.tif",
		                 "p_B4.tif","p_B5.tif","p_B6.tif"};
	string Img_Path2[6] = {"p_B1_250.tif","p_B2_250.tif","p_B3_250.tif",
		                 "p_B4_250.tif","p_B5_250.tif","p_B6_250.tif"};
	string Img_Path3[6] = {"p_B1_500.tif","p_B2_500.tif","p_B3_500.tif",
	                     "p_B4_500.tif","p_B5_500.tif","p_B6_500.tif"};
	for(int i =0; i < 6; i++){
		Ori_Img[i] = imread(Img_Path[i],IMREAD_GRAYSCALE);
	}

	int rows = Ori_Img[0].rows;
	int cols = Ori_Img[0].cols;
	n_Sample = rows * cols;
}
 
//���������ó�ʼ����
void isodata::set_paras()
{
	K = 6;
	theta_c = 5;
	theta_s = 1;
	thetaN = 0.05 * n_Sample;
	maxiteration = 5;
	maxcombine = 2;
	alpha = 0.3;
	current_iter = 0;

	t_assign = 0.0;
	t_update_cen = 0.0;
	t_cacul_mean = 0.0;
	t_cacul_std = 0.0;
	t_split = 0.0;
	t_merge = 0.0;
	n_threads = 4;
}
 
//������������գ�׼����һ�ε���
void isodata::prepare_for_next_itration()
{
	for (int i = 0; i < clus.size(); i++)
		clus[i].clusterID.clear();
}
 
void isodata::apply()
{
	cout<<"��һ����ȷ����ʼ�������ġ�����������"<<endl;
	init();
	while (current_iter < maxiteration)
	{
		current_iter++;
		cout<<" -------------------------------�� "<<current_iter<<" �ε���------------------------------------"<<endl;
		cout<<"�ڶ��������ࡤ��������������"<<endl;
		assign();
		cout<<"��������ȡ��������̫�ٵ��ࡤ��������������"<<endl;
		check_thetaN();
		cout<<"���Ĳ��������������ġ���������������"<<endl;
		update_centers();
		cout<<"���塢����������ƽ�����롤��������������"<<endl;
		calmeandis();
		cout<<"���߲����б���һ�����з��ѻ�ϲ���������㡤��������������"<<endl;
		choose_nextstep();
		if (current_iter < maxiteration)
			prepare_for_next_itration();
	}
	show_result();
}
 
double isodata::distance(const Centroid &cen, const int k)
{
	double dis = 0;
	int r,c;
	r = c =0;
	for (int i = 0; i < dim; i++){
		Id2RC(r,c,k);
		dis += pow(cen[i] -  (int)Ori_Img[i].at<uchar>(r,c), 2);
	}	
	return sqrt(dis);
}
 
double isodata::distance(const Centroid &center1, const Centroid& center2)
{
	double dis = 0;
	for (int i = 0; i < dim; i++)
		dis += pow(center1[i] - center2[i], 2);
	return sqrt(dis);
}

void isodata::Id2RC(int &r, int &c, int id){
	int rows = Ori_Img[0].rows;
	int cols = Ori_Img[0].cols;
	r = id / cols;
	c = id - r * cols;
}
//��һ����ԤѡNc����ʼ��������
void isodata::init()
{
	clus.resize(K);
	int r,c;
	r = c =0;
	set<int>aa;
	for (int i = 0; i < K; i++)
	{
		clus[i].center.resize(dim);
		int id = (i + 0.1)/ (K * 1.0) * n_Sample;
		aa.insert(id);
		for (int j = 0; j < dim; j++){
			 Id2RC(r,c,id);
			 clus[i].center[j] = (int)Ori_Img[j].at<uchar>(r,c);
			 //cout<<clus[i].center[j]<<"  ";
		}
		//cout<<endl;
	}
	int a;
}
 
/*�ڶ�������N��ģʽ�����ָ�����ľ���Sj */
void isodata::assign()
{
	clock_t start;
	clock_t end;
	double duration;
	start = clock();
#pragma omp parallel for num_threads(n_threads)
	for (int i = 0; i < n_Sample; i++)
	{
		double mindis = 100000000;
		int th = -1;
		for (int j = 0; j < clus.size(); j++)
		{
			double dis = distance(clus[j].center, i);
			if (dis < mindis)
			{
				mindis = dis;
				th = j;
			}
		}
	#pragma omp critical
		{
			clus[th].clusterID.push_back(i);
		}		
	}
	for(int i = 0; i < clus.size(); i++){
		cout<<"  �� "<<i+1<<" ������������"<<clus[i].clusterID.size()<<endl;
	}
	end = clock();
	duration =  (double)(end - start) / CLOCKS_PER_SEC;
	t_assign += duration;
	printf("  ���ξ����ʱ��%lf Sec \n", duration); 
}
 
/*�����������Sj�е�������ĿSj<��N��
��ȡ���������Ӽ�����ʱNc��ȥ1*/
void isodata::check_thetaN()
{
	vector<int>toerase;
	for (int i = 0; i < clus.size(); i++)
	{
		if (clus[i].clusterID.size() < thetaN)
		{
			toerase.push_back(i);
			for (int j = 0; j < clus[i].clusterID.size(); j++)
			{
				double mindis = 10000000;
				int th = -1;
				for (int m = 0; m < clus.size(); m++)
				{
					if (m == i)
						continue;
					double dis = distance(clus[m].center,
						clus[i].clusterID[j]);
					if (dis < mindis)
					{
						mindis = dis;
						th = m;
					}
				}
				clus[th].clusterID.push_back(
					clus[i].clusterID[j]);
			}
			clus[i].clusterID.clear();
		}
	}
	for (vector<Cluster>::iterator it = clus.begin(); it != clus.end();)
	{
		if (it->clusterID.empty())
			it = clus.erase(it);
		else
			it++;
	}
}
//���������µ�����������
void isodata::update_center(Cluster &aa)
{
	Centroid temp_F;
	temp_F.resize(dim);
	Centroid *temp = new Centroid[n_threads];
	for(int i = 0; i < n_threads; i++)
		temp[i].resize(dim);
	
#pragma omp parallel for num_threads(n_threads)
	for (int j = 0; j < aa.clusterID.size(); j++)
	{
		int r = 0, c = 0;
		int num = omp_get_thread_num();   //��õ�ǰ�̺߳�
		for (int m = 0; m < dim; m++){
				Id2RC(r,c,aa.clusterID[j]);
			    temp[num][m] += (int)Ori_Img[m].at<uchar>(r,c);
			}			
	}
	for (int m = 0; m < dim; m++){
		for(int i = 0; i < n_threads; i++){
			temp_F[m] += temp[i][m];
		}
		temp_F[m] /= aa.clusterID.size();
	}
	aa.center = temp_F;
}
 
/*���Ĳ�����������������*/
void isodata::update_centers()
{
	clock_t start;
	clock_t end;
	double duration;
	start = clock();
	for (int i = 0; i < clus.size(); i++)
	{
		update_center(clus[i]);
	}
	end = clock();
	duration =  (double)(end - start) / CLOCKS_PER_SEC;
	t_update_cen += duration;
	printf("  ���������������ĺ�ʱ��%lf Sec \n", duration); 
}

//���������µ�������ı�׼��
void isodata::update_sigma(Cluster&bb)
{
	bb.sigma.clear();
	bb.sigma.resize(dim);

	Centroid *sigma_th = new Centroid[n_threads];
	for(int i = 0; i < n_threads; i++)
		sigma_th[i].resize(dim);

#pragma omp parallel for num_threads(n_threads)
	for (int j = 0; j < bb.clusterID.size(); j++){
		int r = 0, c = 0;
		for (int m = 0; m < dim; m++){
			    Id2RC(r,c,bb.clusterID[j]);
				int num = omp_get_thread_num();
				sigma_th[num][m] += pow(bb.center[m] - (int)Ori_Img[m].at<uchar>(r,c), 2);
		}
	}
	
	for (int m = 0; m < dim; m++){
		for(int i = 0; i < n_threads; i++){
			bb.sigma[m] += sigma_th[i][m];
		}
		bb.sigma[m] = sqrt(bb.sigma[m] / bb.clusterID.size());
		//cout<<bb.sigma[m]<<"  ";
	}

}
 
/*�������ϲ�*/
/*���岽�������������Sj��ģʽ��������������ļ��ƽ������*/
/*������������ȫ��ģʽ���������Ӧ�������ĵ���ƽ������*/
void isodata::calmeandis()
{
	clock_t start;
	clock_t end;
	double duration;
	start = clock();
	meandis = 0;
	for (int i = 0; i < clus.size(); i++)
	{
		double dis = 0.0;
		double *dis_t = new double[n_threads];
		for( int n = 0; n < n_threads; n++){
			dis_t[n] = 0;
		}
		#pragma omp parallel for num_threads(n_threads)
		for (int j = 0; j < clus[i].clusterID.size(); j++)
		{
			int num_t = omp_get_thread_num();
			dis_t[num_t] += distance(clus[i].center,clus[i].clusterID[j]);
		}
		for( int n = 0; n < n_threads; n++){
			dis += dis_t[n];
		}
		clus[i].inner_meandis = dis /clus[i].clusterID.size();
		meandis += dis;
	}
	meandis /= n_Sample;
	end = clock();
	duration =  (double)(end - start) / CLOCKS_PER_SEC;
	t_cacul_mean += duration;
	printf("  3.����ƽ�������ʱ��%lf Sec \n", duration); 
	//cout<<"ƽ������"<<meandis<<endl;
}
 
 
/*���߲����б���һ�����з��ѻ�ϲ����������*/
void isodata::choose_nextstep()
{
	if (current_iter == maxiteration)
	{
		theta_c = 0;
		//goto step 11
		check_for_merge();
	}
	else if (clus.size() < K / 2)
	{
		check_for_split();
	}
	else if (current_iter % 2 == 0 ||
		clus.size() >= 2 * K)
	{
		check_for_merge();
	}
	else
	{
		check_for_split();
	}
 
}
/*�ˡ��š�ʮ���ϲ�Ϊ���Ѳ���*/
/*�ڰ˲�������ÿ����������������ı�׼������*/
/*�ھŲ�����ÿһ��׼������{��j, j = 1, 2, ��,
Nc}�е�������*/
/*��ʮ��������*/
void isodata::check_for_split()
{
	cout<<"�ڰˡ��š�ʮ�������ѡ���������������"<<endl;
	clock_t start;
	clock_t end;
	double duration;
	start = clock();
	for (int i = 0; i < clus.size(); i++)
	{
		update_sigma(clus[i]);
	}
	end = clock();
	duration =  (double)(end - start) / CLOCKS_PER_SEC;
	t_cacul_std += duration;
	printf("  ���μ����׼���ʱ��%lf Sec \n", duration); 

	clock_t start2;
	clock_t end2;
	double duration2;
	start2 = clock();
	bool IsSplit_flag = false;
	int cur_size = clus.size();
//#pragma omp parallel for num_threads(n_threads)
		for (int i = 0; i < cur_size; i++)
		{
			int num = 0;
			for (int j = 0; j < dim; j++)
			{
				if (clus[i].sigma[j] > theta_s &&
					( clus[i].inner_meandis > meandis&&clus[i].clusterID.size() > 2 * (thetaN + 1) || clus.size() < K / 2 ))
				{
					IsSplit_flag = true;
					num++;
					split(i);
					cout<<"  �� "<<i+1<<" ������� "<<endl;
					break;
				}
			}
		}
	end2 = clock();
	duration2 =  (double)(end2 - start2) / CLOCKS_PER_SEC;
	t_split += duration2;
	printf("  ���η��Ѻ�ʱ��%lf Sec \n", duration2);
	if( IsSplit_flag = false ){
		check_for_merge();
	}
}
 
void isodata::split(const int kk)
{
	Cluster newcluster;
	newcluster.center.resize(dim);
 
	int th = -1;
	double maxval = 0;
	for (int i = 0; i < dim; i++)
	{
		if (clus[kk].sigma[i] > maxval)
		{
			maxval = clus[kk].sigma[i];
			th = i;
		}
	}
	for (int i = 0; i < dim; i++)
	{
		newcluster.center[i] = clus[kk].center[i];
	}
	newcluster.center[th] -= alpha*clus[kk].sigma[th];
	clus[kk].center[th] += alpha*clus[kk].sigma[th];
	for (int i = 0; i < clus[kk].clusterID.size(); i++)
	{
		double d1 = distance(clus[kk].center, clus[kk].clusterID[i]);
		double d2 = distance(newcluster.center, clus[kk].clusterID[i]);
			if (d2 < d1)
			newcluster.clusterID.push_back(clus[kk].clusterID[i]);
	}
	vector<int>cc; cc.reserve(clus[kk].clusterID.size());
	vector<int>aa;
	int l = clus[kk].clusterID.size() - newcluster.clusterID.size();
	std::sort(clus[kk].clusterID.begin(), clus[kk].clusterID.end());
    std::sort(newcluster.clusterID.begin(), newcluster.clusterID.end());
	//insert_iterator<set<int, less<int> > >res_ins(aa, aa.begin()); 
	set_difference(clus[kk].clusterID.begin(), clus[kk].clusterID.end(),
		newcluster.clusterID.begin(), newcluster.clusterID.end(), inserter(aa, aa.begin()));//�
	clus[kk].clusterID = aa;
	//Ӧ�ø���meandis sigma������
	update_center(newcluster);
	//update_sigma(newcluster);
	update_center(clus[kk]);
	//update_sigma(clus[kk]);
	clus.push_back(newcluster);
}
 
/*��ʮһ��������ȫ���������ĵľ���*/
/*��ʮ�������Ƚ�Dij ���c ��ֵ����Dij <��c ��ֵ����С��������������*/
/*��ʮ������������Ϊ �������������� �� �ϲ�*/
void isodata::check_for_merge()
{
	clock_t start;
	clock_t end;
	double duration;
	start = clock();
	cout<<"��ʮһ��ʮ����ʮ�������ϲ�����������������"<<endl;
	vector<pair<pair<int, int>, double>>aa;
	for (int i = 0; i < clus.size(); i++)
	{
		for (int j = i + 1; j < clus.size(); j++)
		{
			double dis = distance(clus[i].center, clus[j].center);
			if (dis < theta_c)
			{
				pair<int, int>bb(i, j);
				aa.push_back(pair<pair<int, int>, double>(bb, dis));
			}
		}
	}
	// ���ú�������ʵ����������    
	struct CompNameEx
	{
		CompNameEx(bool asce) : asce_(asce)
		{}
		bool operator()(pair<pair<int, int>, double>const& pl, pair<pair<int, int>, double>const& pr)
		{
			return asce_ ? pl.second < pr.second : pr.second < pl.second; // ��Eff STL������21: ��Զ�ñȽϺ�������ȵ�ֵ����false    
		}
	private:
		bool asce_;
	};
	sort(aa.begin(), aa.end(), CompNameEx(true));
	set<int>bb;
	int combinenus = 0;
	for (int i = 0; i < aa.size(); i++)
	{
		if (bb.find(aa[i].first.first) == bb.end()
			&& bb.find(aa[i].first.second) == bb.end())
		{
			bb.insert(aa[i].first.first);
			bb.insert(aa[i].first.second);
			merge(aa[i].first.first, aa[i].first.second);
			combinenus++;
			if (combinenus >= maxcombine)
				break;
		}
	}
	for (vector<Cluster>::iterator it = clus.begin(); it != clus.end();)
	{
		if (it->clusterID.empty())
		{
			it = clus.erase(it);
		}
		else
			it++;
	}

	end = clock();
	duration =  (double)(end - start) / CLOCKS_PER_SEC;
	t_merge += duration;
	printf("  7.�ϲ���ʱ��%lf Sec \n", duration); 
}
 
void isodata::merge(const int k1, const int k2)//k1��k2˳���ܱ�
{
	for (int i = 0; i < dim; i++)
		clus[k1].center[i] = (clus[k1].center[i] * clus[k1].clusterID.size() +
		clus[k2].center[i] * clus[k2].clusterID.size()) /
		double(clus[k1].clusterID.size() + clus[k2].clusterID.size());
	//clus[k1].clusterID.insert(clus[k1].clusterID.end(), 
	//	clus[k2].clusterID.begin(), clus[k2].clusterID.end());
	clus[k2].clusterID.clear();
}


int _tmain(int argc, _TCHAR* argv[])
{
	clock_t start;
	clock_t end;
	double duration;
	start = clock();

	isodata iso;
	iso.generate_data();
	iso.set_paras();
	iso.apply();

	end = clock();
	duration =  (double)(end - start) / CLOCKS_PER_SEC;
	printf("����ʱ�䣺%lf Sec ", duration); 

	system("pause");
	return 0;
}
