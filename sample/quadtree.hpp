#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <random>
#include <limits>
#include <cassert>

#include "../utility.h"
#include "../CSR.h"
#include "../multiply.h"

#include "../sample/commonutility.hpp"
#include "../sample/Coordinate.hpp"
using namespace std;

#define VALUETYPE double
#define INDEXTYPE int
#define MAXMIN 15.0
#define t 0.999

class Node{
    Coordinate pos;
    float data;
    double force_x;
    double force_y;
};

class quadtree{
    Coordinate topLeft;
    Coordinate botRight;
    float center_of_mass_x;
    float center_of_mass_y;
    float total_mass;
    float number_of_particles;
    Node *n;
    quadtree *topLeftTree;
    quadtree *topRightTree;
    quadtree *botLeftTree;
    quadtree *botRightTree;
};
void quadtreeInit(quadtree* treenode, Coordinate tL, Coordinate bR){
    treenode->topLeft=tL;
    treenode->botRight=bR;
    treenode->n=NULL;
    treenode->topLeftTree  = NULL;
    treenode->topRightTree = NULL;
    treenode->botLeftTree  = NULL;
    treenode->botRightTree = NULL;
    treenode->center_of_mass_x = 0;
    treenode->center_of_mass_y = 0;
    treenode->total_mass = 0;
    treenode->number_of_particles = 0;
}

void insert( quadtree* treenode,  Node* node){
    if (node == NULL)
        return;
    if (!inBoundary(treenode, node->pos))
        return;
    if (abs(treenode->topLeft.x - treenode->botRight.x) <= 1 && abs(treenode->topLeft.y - treenode->botRight.y) <= 1){
        if (treenode->n == NULL)
            treenode->n = node;
        return;
    }
 
    if ((treenode->topLeft.x + treenode->botRight.x) / 2 >= node->pos.x){
        if ((treenode->topLeft.y + treenode->botRight.y) / 2 >= node->pos.y){
            if (treenode->topLeftTree == NULL)
            {
				treenode->topLeftTree = ( quadtree*)malloc(sizeof( quadtree));
                 Coordinate tL = {treenode->topLeft.x, treenode->topLeft.y};
                 Coordinate bR = {(treenode->topLeft.x+treenode->botRight.x)/2,(treenode->topLeft.y+treenode->botRight.y)/2};
				quadtreeInit(treenode->topLeftTree,tL,bR);
        	}   
            insert(treenode->topLeftTree,node);
        }else{
            if (treenode->botLeftTree == NULL){
            	treenode->botLeftTree = ( quadtree*)malloc(sizeof( quadtree));
                 Coordinate tL = {treenode->topLeft.x, (treenode->topLeft.y+treenode->botRight.y)/2};
                 Coordinate bR = {(treenode->topLeft.x+treenode->botRight.x)/2,treenode->botRight.y};
                quadtreeInit(treenode->botLeftTree,tL,bR);
			}
            insert(treenode->botLeftTree,node);
        }
    }else{
        if ((treenode->topLeft.y + treenode->botRight.y) / 2 >= node->pos.y)
        {
            if (treenode->topRightTree == NULL)
            {
            	treenode->topRightTree = ( quadtree*)malloc(sizeof( quadtree));
                 Coordinate tL = {(treenode->topLeft.x+treenode->botRight.x)/2, treenode->topLeft.y};
                 Coordinate bR = {treenode->botRight.x,(treenode->topLeft.y+treenode->botRight.y)/2};
                quadtreeInit(treenode->topRightTree,tL,bR);            	
			}
            insert(treenode->topRightTree,node);
        }
 
        else
        {
            if (treenode->botRightTree == NULL)
            {
            	treenode->botRightTree = ( quadtree*)malloc(sizeof( quadtree));
                 Coordinate tL = {(treenode->topLeft.x+treenode->botRight.x)/2, (treenode->topLeft.y+treenode->botRight.y)/2};
                 Coordinate bR = {treenode->botRight.x,treenode->botRight.y};
                quadtreeInit(treenode->botRightTree,tL,bR);
			}
			insert(treenode->botRightTree,node);
        }
    }
}

void initialize_Coordinate( Coordinate P, float x1 , float y1){
	printf("%f%f\n",x1,y1);
	P.x=x1;
	P.y=y1;
}

void initialize_Node( Node* node,  Coordinate P, float value){
	node->pos=P;
	node->data=value;
	node->force_x=0;
	node->force_y=0;
}

 Node* search( quadtree* treenode,  Coordinate p){
    if (!inBoundary(treenode, p))
        return NULL;
    if (treenode->n != NULL)
        return treenode->n;
	
    if ((treenode->topLeft.x + treenode->botRight.x) / 2 >= p.x)
    {
        if ((treenode->topLeft.y + treenode->botRight.y) / 2 >= p.y)
        {
            if (treenode->topLeftTree == NULL)
                return NULL;
            return search(treenode->topLeftTree, p);
        }
        else
        {
            if (treenode->botLeftTree == NULL)
                return NULL;
            return search(treenode->botLeftTree, p);
        }
    }
    else
    {
        if ((treenode->topLeft.y + treenode->botRight.y) / 2 >= p.y)
        {
            if (treenode->topRightTree == NULL)
                return NULL;
            return search(treenode->topRightTree, p);
        }
        else
        {
            if (treenode->botRightTree == NULL)
                return NULL;
            return search(treenode->botRightTree, p);
        }
    }
}

bool inBoundary( quadtree* treenode,  Coordinate p){
    return (p.x >= treenode->topLeft.x &&
        p.x <= treenode->botRight.x &&
        p.y >= treenode->topLeft.y &&
        p.y <= treenode->botRight.y);
}

void calc_center_of_mass( quadtree* treenode){
	if(treenode->n!=NULL)
	{
		treenode->center_of_mass_x=treenode->n->pos.x;
		treenode->center_of_mass_y=treenode->n->pos.y;
		treenode->total_mass=treenode->n->data;
		treenode->number_of_particles=1;
	}
	else
	{
		if(treenode->topLeftTree!=NULL)
		{
			calc_center_of_mass(treenode->topLeftTree);
			treenode->total_mass+=(float)treenode->topLeftTree->total_mass;
			treenode->center_of_mass_x+=(float)(treenode->topLeftTree->center_of_mass_x*treenode->topLeftTree->total_mass);
			treenode->center_of_mass_y+=(float)(treenode->topLeftTree->center_of_mass_y*treenode->topLeftTree->total_mass);
			treenode->number_of_particles+=(float)treenode->topLeftTree->number_of_particles;
		}
		if(treenode->botLeftTree!=NULL)
		{
			calc_center_of_mass(treenode->botLeftTree);
			treenode->total_mass+=(float)treenode->botLeftTree->total_mass;
			treenode->center_of_mass_x+=(float)(treenode->botLeftTree->center_of_mass_x*treenode->botLeftTree->total_mass);
			treenode->center_of_mass_y+=(float)(treenode->botLeftTree->center_of_mass_y*treenode->botLeftTree->total_mass);
			treenode->number_of_particles+=(float)treenode->botLeftTree->number_of_particles;
		}
		if(treenode->topRightTree!=NULL)
		{
			calc_center_of_mass(treenode->topRightTree);
			treenode->total_mass+=(float)treenode->topRightTree->total_mass;
			treenode->center_of_mass_x+=(float)(treenode->topRightTree->center_of_mass_x*treenode->topRightTree->total_mass);
			treenode->center_of_mass_y+=(float)(treenode->topRightTree->center_of_mass_y*treenode->topRightTree->total_mass);
			treenode->number_of_particles+=(float)treenode->topRightTree->number_of_particles;
		}
		if(treenode->botRightTree!=NULL)
		{
			calc_center_of_mass(treenode->botRightTree);
			treenode->total_mass+=(float)treenode->botRightTree->total_mass;
			treenode->center_of_mass_x+=(float)(treenode->botRightTree->center_of_mass_x*treenode->botRightTree->total_mass);
			treenode->center_of_mass_y+=(float)(treenode->botRightTree->center_of_mass_y*treenode->botRightTree->total_mass);
			treenode->number_of_particles+=(float)treenode->botRightTree->number_of_particles;
		}
		treenode->center_of_mass_x=treenode->center_of_mass_x/treenode->total_mass;
		treenode->center_of_mass_y=treenode->center_of_mass_y/treenode->total_mass;
	}
	
	printf("This section has \n");
	printf("Center of Mass x : %f\n",treenode->center_of_mass_x);
	printf("Center of Mass y : %f\n",treenode->center_of_mass_y);
	printf("Total Mass : %f\n",treenode->total_mass);
	printf("Total Particles : %f\n\n",treenode->number_of_particles);
}

void calc_force( Node** node,  quadtree* treenode, int num){
	int i=0;

	double start=omp_get_wtime();	
	#pragma omp parallel for schedule(guided)
	for(i=0;i<num;i++)
	{
		calc_node_force(node[i], treenode);
		printf("Force x for node(%0.2f,%0.2f) of mass %0.2f : %0.10lf *G N\n",node[i]->pos.x,node[i]->pos.y, node[i]->data, node[i]->force_x);
		printf("Force y for node(%0.2f,%0.2f) of mass %0.2f : %0.10lf *G N\n\n",node[i]->pos.x,node[i]->pos.y, node[i]->data, node[i]->force_y);
	}
	double end2 = omp_get_wtime();
	double time = end2-start;
	printf("\nTime : %0.10lf\n", time);
}

void calc_node_force( Node* node,  quadtree* treenode){	
	if(treenode->number_of_particles==1)
	{
		force_of_2_body(node, treenode);
	}
	else
	{
		double x1=(double)node->pos.x;
		double y1=(double)node->pos.y;
		double x2=(double)treenode->center_of_mass_x;
		double y2=(double)treenode->center_of_mass_y;
		double r=sqrt(pow(x2-x1,2)+pow(y2-y1,2));
		double d=treenode->botRight.x-treenode->topLeft.x;
		double theta = 0.5;
		if(d/r<theta)
		{
			force_of_2_body(node, treenode);
		}
		else
		{
			if(treenode->topLeftTree!=NULL)
				calc_node_force(node, treenode->topLeftTree);
			if(treenode->botLeftTree!=NULL)
				calc_node_force(node, treenode->botLeftTree);
			if(treenode->topRightTree!=NULL)
				calc_node_force(node, treenode->topRightTree);
			if(treenode->botRightTree!=NULL)
				calc_node_force(node, treenode->botRightTree);
		}
	}
}

void force_of_2_body( Node* node,  quadtree* treenode){
	double G = 1;//6.67*pow(10,-6);
	double x1=(double)node->pos.x;
	double y1=(double)node->pos.y;
	double x2=(double)treenode->center_of_mass_x;
	double y2=(double)treenode->center_of_mass_y;
	double m1=(double)node->data;
	double m2=(double)treenode->total_mass;
	double r=sqrt(pow(x2-x1,2)+pow(y2-y1,2));
	if(r==0)
	{
		return;
	}
	node->force_x += (x2-x1)*G*m1*m2/(r*r*r);
	node->force_y += (y2-y1)*G*m1*m2/(r*r*r);
}
