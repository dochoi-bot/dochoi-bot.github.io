#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <memory.h>
#include <utility>
#include <unordered_set>
#include <cmath>
#include<random>
#include <stdexcept>
using namespace std;

/**
 * Grab Snaffles and try to throw them through the opponent's goal!
 * Move towards a Snaffle to grab it and use your team id to determine towards where you need to throw it.
 * Use the Wingardium spell to move things around at your leisure, the more magic you put it, the further they'll move.
 **/
 
 /**
  * 
  * radar value 0 == empty 
  * 1 == snaf 
  * 2 == wiz 
  * 3 == buz 
  * 4 == owiz
  * 6 7 8
  * 5   1
  * 4 3 2 
  **/
#define RAD2DEG 57.2957951
#define FRICTION_WIZARD 0.75
#define FRICTION_BLUDEGER 0.9
#define FRICTION_SNAFFLES 0.75

#define MASS_WIZARD 1.0
#define MASS_SNAFFLES 0.5
#define MASS_BLUDEGER 8.0

#define RADIUS_WIZARD 400
#define RADIUS_BLUDEGER 200
#define RADIUS_SNAFFLES 150
int g_dx[9] = {0,1000,1000,0,-1000,-1000,-1000,0,1000};
int g_dy[9] = {0,0,1000,1000,1000,0,-1000,-1000,-1000};
typedef struct s_node{
    
            int entityId; // entity identifier
            string entityType; // "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" or "BLUDGER"
            int x; // position
            int y; // position
            int vx; // velocity
            int vy; // velocity
            int state;
            
            double d_ux;
            double d_uy;
            double d_norm_vector;
            int direction;
            double d_mass;
            double d_force;
            int nx;
            int ny;
            int check;
            s_node *radar[3][3];

}t_node;
int g_myTeamId;
t_node* mygoal_node;
t_node* enemygoal_node;
vector<t_node*> snaf;
vector<t_node*> wiz;
vector<t_node*> owiz;
vector<t_node*> bger;
vector<t_node*> col;
bool possible_commend;
bool use_magic;
int magic = 10;

void find_radar2(t_node* node, vector<t_node*> v)
{
    int standard = 500;
    int base_x = node->nx;
    int base_y = node->ny;
      
    for (int i = 0; i < v.size(); i++)
    {
        if (v[i]->entityId == node->entityId)
            continue;
        int cal_x = v[i]->nx - base_x;
        int cal_y = v[i]->ny - base_y;
        if (cal_x >= -(3 * standard) && cal_x < -(standard))
        {
            if (cal_y >=-(3 * standard) && cal_y < -(standard))
                node->radar[0][0] = v[i];
            else if(cal_y >= -(standard) && cal_y < (standard))
                node->radar[1][0] = v[i];
            else if (cal_y >= (standard) && cal_y <= (3 * standard))
                node->radar[2][0] = v[i];
        }
        else if (cal_x >= -(standard) && cal_x < (standard))
        {
            if (cal_y >= -(3 * standard) && cal_y < -(standard))
                node->radar[0][1] = v[i];
            else if(cal_y >= -(standard) && cal_y < (standard))
                node->radar[1][1] = v[i];
            else if (cal_y >= (standard) && cal_y <= (3 * standard))
                node->radar[2][1] = v[i];
        }
        else if (cal_x >= (standard) && cal_x <= (3 * standard))
        {
            if (cal_y >=-(3 * standard) && cal_y < -(standard))
        {
        node->radar[0][2] = v[i];
        }
            else if(cal_y >= -(standard) && cal_y < (standard))
                node->radar[1][2] = v[i];
            else if (cal_y >= (standard)  && cal_y <= (3 * standard))
                node->radar[2][2] = v[i];
        }
        
    }
}


void find_radar(t_node* node)
{
find_radar2(node, snaf);
find_radar2(node, wiz);
find_radar2(node, owiz);
find_radar2(node, col);
find_radar2(node, bger);

}
int calculate_direction(t_node* node, t_node* node2) // using ny nx
{
    int direction = 0;
    int y = node2->ny - node->ny;
    int x = node2->nx - node->nx;
    if (y ==0 && x == 0)
        return (-1);
 double angle =  atan((double)(abs(y)) / (double)(abs(x)));
 angle *= RAD2DEG;
 
    double norm = sqrt(((y) * (y)) + ((x) * (x)));
    double uy = (double)((y)) / norm;
    double ux = (double)((x)) / norm;
    
 if ((y) > 0 && (x) < 0)
    angle = 180 - angle;
 else if  ((y) < 0 && (x) < 0)
    angle += 180;
 else if  ((y) < 0 && (x) > 0)
    angle = 360 - angle;
    
    if (angle > 337.5 || angle <= 22.5)
        direction = 1;
    else if (angle > 22.5 && angle <= 67.5)
        direction = 2;
    else if (angle > 67.5 && angle <= 112.5)
        direction = 3;
    else if (angle > 112.5 && angle <= 157.5)
        direction = 4;
    else if (angle > 157.5 && angle <= 202.5)
        direction = 5;
    else if (angle > 202.5 && angle <= 247.5)
        direction = 6;
    else if (angle > 247.5 && angle <= 292.5)
        direction = 7;
    else if (angle > 292.5 && angle <= 337.5)
        direction = 8;
    return (direction);
}
void calculate_direction_and_vector(t_node* node)
{
    
 double angle =  atan((double)(abs(node->vy)) / (double)(abs(node->vx)));
 angle *= RAD2DEG;
 
    double norm = sqrt((node->vy * node->vy) + (node->vx * node->vx));
    double uy = (double)(node->vy) / norm;
    double ux = (double)(node->vx) / norm;
    node->d_uy = uy;
    node->d_ux = ux;
    node->d_norm_vector = sqrt((uy * uy) + (ux * ux));
    if (node->vx == 0 && node->vy == 0)
        {
            node->ny = node->y;
            node->nx = node->x;
            node->d_uy = 0;
            node->d_ux = 0;
            node->d_norm_vector = 0;
            node->direction = -1;
            return ;
        }
 if (node->vy > 0 && node->vx < 0)
    angle = 180 - angle;
 else if  (node->vy < 0 && node->vx < 0)
    angle += 180;
 else if  (node->vy < 0 && node->vx > 0)
    angle = 360 - angle;
    
    if (angle > 337.5 || angle <= 22.5)
        node->direction = 1;
    else if (angle > 22.5 && angle <= 67.5)
        node->direction = 2;
    else if (angle > 67.5 && angle <= 112.5)
        node->direction = 3;
    else if (angle > 112.5 && angle <= 157.5)
        node->direction = 4;
    else if (angle > 157.5 && angle <= 202.5)
        node->direction = 5;
    else if (angle > 202.5 && angle <= 247.5)
        node->direction = 6;
    else if (angle > 247.5 && angle <= 292.5)
        node->direction = 7;
    else if (angle > 292.5 && angle <= 337.5)
        node->direction = 8;
}
void calculate_next(t_node* node)
{
      
    node->nx = node->x  + (node->vx) + (int)(node->d_ux * (node->d_force / node->d_mass));
    node->ny = node->y  + (node->vy) + (int)(node->d_uy * (node->d_force / node->d_mass));
}

double calculate_dist(t_node* node, t_node* node2)
{
    int y = node->ny - node2->ny;
    int x = node->nx - node2->nx;
    
    double dist = sqrt((y * y) + (x * x));
    return (dist);
}
void set_col()
{
    t_node* st3 = new t_node;
    t_node* st4 = new t_node;
    t_node* st5 = new t_node;
    t_node* st6 = new t_node;
        memset(st3, 0, sizeof(t_node));
            memset(st4, 0, sizeof(t_node));
                memset(st5, 0, sizeof(t_node));
                    memset(st6, 0, sizeof(t_node));
    st3->nx = st3->x = 0;
   st3->ny =  st3->y = 5750;
    st3->d_mass = 10000;
    st4->nx = st4->x = 0;
    st4->d_mass = 10000;
    st4->ny =st4->y = 1750;
    st5->nx =st5->x = 16000;
    st5->d_mass = 10000;
    st5->ny =st5->y = 5750;
    st6->nx = st6->x = 16000;
    st6->ny = st6->y = 1750;
    st6->d_mass = 10000;

  st3->entityId = -10;
   st4->entityId = -10;
    st5->entityId = -10;
     st6->entityId = -10;
     
    st3->entityType == "COLx";
    st4->entityType == "COLx";
    st5->entityType == "COLx";
    st6->entityType == "COLx";
    
    col.push_back(st3);
     col.push_back(st4);
      col.push_back(st5);
       col.push_back(st6);
}

t_node *direction_to_radar(t_node *mywiz, int direction)
{
    if (direction == 1)
        return (mywiz->radar[1][2]);
     if (direction == 2)
        return (mywiz->radar[2][2]);
         if (direction == 3)
        return (mywiz->radar[2][1]);
         if (direction == 4)
        return (mywiz->radar[2][0]);
         if (direction == 5)
        return (mywiz->radar[1][0]);
         if (direction == 6)
        return (mywiz->radar[0][0]);
         if (direction == 7)
        return (mywiz->radar[0][1]);
         if (direction == 8)
        return (mywiz->radar[0][2]);
    
    return (0);
}


void move(t_node *mywiz, t_node *target)
{   
    
  int direction = mywiz->direction;
    //int direction = calculate_direction(mywiz, target);
        cerr<<"[ "<< direction <<"]"<< endl;
    if (direction == -1)
        return ;
        for (int i = -1; i < 2; i++)
        {
            int w_direction = direction + i;
            if (w_direction == 0)
                w_direction = 8;
            if (w_direction == 9)
                w_direction = 1;
            
            t_node *who =  direction_to_radar(mywiz, w_direction);
            if (who != 0)
            {
                if (who->entityType == "BLUDGER")
                {
                    //cerr << who->d_ux <<" " << who->d_uy<<endl;
                    // cerr << mywiz->d_ux <<" "<< mywiz->d_uy<<endl;
                 
                    double ux_sum = who->d_ux + mywiz->d_ux;
                    double uy_sum = who->d_uy + mywiz->d_uy;
                        // cerr << ux_sum <<" "<< uy_sum<<endl;
                    ux_sum *= 100000;
                    uy_sum *= 100000;
                    cout << "MOVE " <<  (int)ux_sum << " " << (int)uy_sum << " 150" << " 슥" <<endl;
                    possible_commend = 0;
                    target->check = 1;
                    return;
                }
            }
        }
  //  cerr << direction<< endl;
    
}

t_node *what_is_close(t_node *mywiz, t_node *partner)
{
    if (snaf.size() == 1)
        return (snaf[0]);
    
    t_node *target = snaf[0];
    
    double min_dist = 9999999999.0;
    for (int i = 0; i < snaf.size(); i++)
    {
        if (snaf[i]->check == 1)
            continue;
        double dist = calculate_dist(mywiz, snaf[i]);
        if (dist < min_dist)
        {
            target = snaf[i];
            min_dist = dist;
        }
    }
    // if (use_magic == 0)
    // {
    //     t_node *target2 = 0;
    //     min_dist = 9999999999.0;
    //     for (int i = 0; i < snaf.size(); i++)
    //     {
    //         if (target == snaf[i])
    //             continue;
    //         if (snaf[i]->check == 1)
    //         continue;
    //         double dist = calculate_dist(mywiz, snaf[i]);
    //         if (dist < min_dist)
    //         {
    //             target2 = snaf[i];
    //             min_dist = dist;
    //         }
    //     }
    //     if (target2 != 0)
    //     {
    //     double dist_my = calculate_dist(mywiz, target) + calculate_dist(partner, target2);
    //     double dist_you = calculate_dist(mywiz, target2) + calculate_dist(partner, target);
    //     if (dist_you < dist_my)
    //         return (target2);
    //     }
    // }
        return (target);

}


void shoot_cal(t_node *mywiz, int x, int y)
{
//      int y = enemygoal_node->ny - mywiz->ny;
  //  int x = enemygoal_node->nx - mywiz->nx;
    double norm = sqrt(((y) * (y)) + ((x) * (x)));
    double target_uy = (double)((y)) / norm;
    double target_ux = (double)((x)) / norm;
    double ux_diff = (1000 * target_ux) - (double)(mywiz->vx); // 500 / 0.5 == 1000; 
    double uy_diff = (1000 * target_uy) - (double)(mywiz->vy);
    //cerr << ux_diff <<" "<< uy_diff<<endl;
            ux_diff *= 100;
            uy_diff *= 100;
        cout << "THROW " <<  (int)ux_diff << " " << (int)uy_diff <<" 500" << endl;
        possible_commend = 0;
}
void shoot_case3(t_node *mywiz, int w_direction1, int w_direction2, t_node *who1, t_node *who2)
{
    if (who1->entityType == "OPPONENT_WIZARD" && who2->entityType != "OPPONENT_WIZARD")
    {
        w_direction2 += 1;
        if (w_direction2 == 9)
            w_direction2 = 1;
        int ny = g_dy[w_direction2];
    int nx =  g_dx[w_direction2];
    shoot_cal(mywiz, nx, ny);
        
    }
    else if (who2->entityType == "OPPONENT_WIZARD" && who1->entityType != "OPPONENT_WIZARD")
    {
        w_direction1 -= 1;
        if (w_direction1 == 0)
            w_direction1 = 8;
        int ny =  g_dy[w_direction1];
        int nx =  g_dx[w_direction1];
    shoot_cal(mywiz, nx, ny);
    }
    else
    {
        for(int i = 0; i < 2; i++)
        {
            if (wiz[i]->entityId != mywiz->entityId)
            {
                int ny =  wiz[!i]->ny  - wiz[i]->ny ;
                int nx = wiz[!i]->nx - wiz[i]->nx ;
                shoot_cal(mywiz, nx, ny);
                return;
            }
        }
    }
}

void shoot_case2(t_node *mywiz, int w_direction)
{
     int ny =g_dy[w_direction];
    int nx = g_dx[w_direction];
    shoot_cal(mywiz, nx, ny);

}
void shoot_case1(t_node *mywiz, int w_direction1, int w_direction2)
{
    int ny1 = mywiz->ny + g_dy[w_direction1];
    int nx1 = mywiz->nx + g_dx[w_direction1];
    
    int ny2 = mywiz->ny + g_dy[w_direction2];
    int nx2 = mywiz->nx + g_dx[w_direction2];
    
     int y1 = ny1 - enemygoal_node->ny;
    int x1 = ny1 - enemygoal_node->nx;
    int y2 = ny2 - enemygoal_node->ny;
    int x2 = ny2 - enemygoal_node->nx;
    
    double dist1 = sqrt((y1 * y1) + (x1 * x1));
    double dist2 = sqrt((y2 * y2) + (x2 * x2));
    if (dist1 < dist2)
    {
        shoot_cal(mywiz, g_dx[w_direction1],  g_dy[w_direction1]);
    }
    else
    {
         shoot_cal(mywiz,g_dx[w_direction2],g_dy[w_direction2]);
    }
}



void shoot(t_node *mywiz)
{
     int direction = calculate_direction(mywiz, enemygoal_node);
    if (direction == -1)
        return ;
        t_node *who =  direction_to_radar(mywiz, direction);
    if (who != 0)
    {
        if (who->entityType == "BLUDGER" || who->entityType == "OPPONENT_WIZARD" || who->entityType == "COL")
        {
          
                int w_direction1 = direction - 1;
                int w_direction2 = direction + 1;
                if (w_direction1 == 0)
                    w_direction1 = 8;
                if (w_direction2 == 9)
                    w_direction2 = 1;
                    
             t_node *who1 = 0;
             t_node *who2 = 0;
             
             who1 =  direction_to_radar(mywiz, w_direction1);
             who2=  direction_to_radar(mywiz, w_direction2);
             if (who1 == 0 && who2 == 0)
             {
                 shoot_case1(mywiz, w_direction1, w_direction2 );
             }
             else if (who1 != 0 && who2 == 0)
             {
                 if (who1->entityType == "BLUDGER" || who1->entityType == "OPPONENT_WIZARD" || who1->entityType == "COL")
                    shoot_case2(mywiz, w_direction2);
                else
                    shoot_case1(mywiz, w_direction1, w_direction2 );
             }
             else if (who1 == 0 && who2 != 0)
             {  
                 if (who2->entityType == "BLUDGER" || who2->entityType == "OPPONENT_WIZARD" || who2->entityType == "COL")
                 shoot_case2(mywiz, w_direction1);
                 else
                    shoot_case1(mywiz, w_direction1, w_direction2 );
             }
             else if (who1 != 0 && who2 != 0)
             { 
                 if (who2->entityType == "BLUDGER" || who2->entityType == "OPPONENT_WIZARD" || who2->entityType == "COL")
                 {
                     if (who1->entityType == "BLUDGER" || who1->entityType == "OPPONENT_WIZARD" || who1->entityType == "COL")
                     {                 
                         shoot_case3(mywiz, w_direction1, w_direction2, who1, who2);
                     }
                     else
                     {
                         shoot_case2(mywiz, w_direction1);
                     }
                }
                else if (who1->entityType == "BLUDGER" || who1->entityType == "OPPONENT_WIZARD" || who1->entityType == "COL")
                    {
                        if (who2->entityType == "BLUDGER" || who2->entityType == "OPPONENT_WIZARD" || who2->entityType == "COL")
                        {                 
                            shoot_case3(mywiz, w_direction1, w_direction2, who1, who2);
                        }
                        else
                        {
                            shoot_case2(mywiz, w_direction2);
                        }

                    }      
            }    
        }
    }
    
    
    if (possible_commend)
    {
    int y = enemygoal_node->ny - mywiz->ny;
    int x = enemygoal_node->nx - mywiz->nx;
    double norm = sqrt(((y) * (y)) + ((x) * (x)));
    double target_uy = (double)((y)) / norm;
    double target_ux = (double)((x)) / norm;
    double ux_diff = (1000 * target_ux) - (double)(mywiz->vx); // 500 / 0.5 == 1000; 
    double uy_diff = (1000 * target_uy) - (double)(mywiz->vy);
  //  cerr << ux_diff <<" "<< uy_diff<<endl;
            ux_diff *= 100;
            uy_diff *= 100;
        cout << "THROW " <<  (int)ux_diff << " " << (int)uy_diff <<" 500" << endl;
        possible_commend = 0;
    }
   // cerr << direction<< endl;
}

void magic_cal(t_node *ball, int tx, int ty, double min_dist)
{
 
    int y = ty - ball->ny;
    int x = tx - ball->nx;
    
    //double mindist = 2500;
    double norm = sqrt(((y) * (y)) + ((x) * (x)));
    double target_uy = (double)((y)) / norm;
    double target_ux = (double)((x)) / norm;
    for(int point = 15 ; point <= 100; point++)
    {

        double ux_diff = ((15 * point / 0.5) * target_ux) - (double)(ball->vx); // 500 / 0.5 == 1000; 
        double uy_diff = ((15 * point / 0.5) * target_uy) - (double)(ball->vy);
        double scala = sqrt((uy_diff * uy_diff) + (ux_diff * ux_diff));

        if (scala > min_dist && magic >= point)
        {
                ux_diff *= 100;
                uy_diff *= 100;
        cout << "WINGARDIUM " << ball->entityId  << " " << (int)ux_diff <<" " << (int)uy_diff << " " << point << endl;
                                    possible_commend = 0;
                                    magic -= point;
            ball->check = 1;
            use_magic = 1;
            return;
        }
    }
}
int who_close(t_node *ball) // //1 is 내가더 가까움0is 상대가 더 가까움
{
    double mindist1 = 9999999999;
    double mindist2 = 9999999999;
    for(int i = 0; i < wiz.size(); i++)
    {
        mindist1= min(calculate_dist(wiz[i], ball), mindist1);
    }
    for(int i = 0; i < owiz.size(); i++)
    {   mindist2= min(calculate_dist(owiz[i], ball), mindist2);
    }
    return (mindist1 < mindist2);
}
void defense()
{

    double dfmax_dist = 2500;
    double rdist = 500;
    if(g_myTeamId == 0)
    {
        int whoisthere[3] = {0};
        int defensex = 5000;
        int defensey[3] = {0, 3750, 7500};
        for(int i = 0; i < owiz.size(); i++)
        {
            if (owiz[i]->nx > 8000)
                continue;
            if (owiz[i]->ny <2500)
                whoisthere[0] = 1;
            if (owiz[i]->ny >= 2500 && owiz[i]->ny < 5000)
                whoisthere[1] = 1;
            else
                whoisthere[2]= 1;
        }
        for(int i = 0; i < snaf.size(); i++)
        {
            if (snaf[i]->check == 1 || who_close(snaf[i]))
                continue;
            for(int ny = 2500; ny <=5000; ny += 1250)
            {
               // cerr <<" AA" <<endl;
                int y = ny - snaf[i]->ny;
                int x =  snaf[i]->nx;
                 //cerr << x << " " <<  y << endl;;
                double dist = sqrt((y * y) + (x * x));
       
                if (dist < dfmax_dist)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        if (whoisthere[j] == 0)
                        {   
                            magic_cal(snaf[i], defensex, defensey[j],dfmax_dist -  dist + rdist);
                            //cout << "WINGARDIUM " << snaf[i]->entityId  << " " << defensex <<" " << defensey[i] << " " << j << endl;

                            return ;
                        }
                    }
                }
            }
        }
    }
    else
    {
        
         int whoisthere[3] = {0};
        int defensex = 11000;
        int defensey[3] = {0, 3750, 7500};
        for(int i = 0; i < owiz.size(); i++)
        {
            if (owiz[i]->nx < 8000)
                continue;
            if (owiz[i]->ny <2500)
                whoisthere[0] = 1;
            if (owiz[i]->ny >= 2500 && owiz[i]->ny < 5000)
                whoisthere[1] = 1;
            else
                whoisthere[2]= 1;
        }
        for(int i = 0; i < snaf.size(); i++)
        {
            if (snaf[i]->check == 1 || who_close(snaf[i]))
                continue;
            for(int ny = 2500; ny <=5000; ny += 1250)
            {
               // cerr <<" AA" <<endl;
                int y = ny - snaf[i]->ny;
                int x =  16000 - snaf[i]->nx;
                 //cerr << x << " " <<  y << endl;;
                double dist = sqrt((y * y) + (x * x));

                if (dist < dfmax_dist)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        if (whoisthere[j] == 0)
                        {   
                            magic_cal(snaf[i], defensex, defensey[j], dfmax_dist -  dist + rdist);
                            //cout << "WINGARDIUM " << snaf[i]->entityId  << " " << defensex <<" " << defensey[i] << " " << j << endl;

                            return ;
                        }
                    }
                }
            }
        }
    }
}




void attack()
{
    double dfmax_dist = 2500;
    if(g_myTeamId == 1)
    {
        int whoisthere[3] = {0};
        int attackx = 2000;
        int attacky[3] = {0, 3750, 7500};
        for(int i = 0; i < owiz.size(); i++)
        {
            if (owiz[i]->nx > attackx)
                continue;
            if (owiz[i]->ny <2500)
                whoisthere[0] = 1;
            if (owiz[i]->ny >= 2500 && owiz[i]->ny < 5000)
                whoisthere[1] = 1;
            else
                whoisthere[2]= 1;
        }
        for(int i = 0; i < snaf.size(); i++)
        {
            if (snaf[i]->check == 1 || who_close(snaf[i]))
                continue;

            for(int ny = 2500; ny <=5000; ny += 1250)
            {
       
                int y = ny - snaf[i]->ny;
                int x =  snaf[i]->nx;
                 
                double dist = sqrt((y * y) + (x * x));
      
                if (dist < dfmax_dist)
                {
                    for(int j = 0; j < 3; j++)
                    {
           
                        
                        if (whoisthere[j] == 0)
                        {   //   cerr <<whoisthere[j] <<"ccc"<< endl;
                            magic_cal(snaf[i], 0, ny, dist);
                            //cout << "WINGARDIUM " << snaf[i]->entityId  << " " << attackx <<" " << attacky[i] << " " << j << endl;

                            return ;
                        }
                    }//cerr <<endl;
                }
            }
        }
    }
    else
    {
        
         int whoisthere[3] = {0};
        int attackx = 14000;
        int attacky[3] = {0, 3750, 7500};
        for(int i = 0; i < owiz.size(); i++)
        {
            if (owiz[i]->nx < attackx)
                continue;
            if (owiz[i]->ny <2500)
                whoisthere[0] = 1;
            if (owiz[i]->ny >= 2500 && owiz[i]->ny < 5000)
                whoisthere[1] = 1;
            else
                whoisthere[2]= 1;
        }
       for(int i = 0; i < snaf.size(); i++)
        {
            if (snaf[i]->check == 1 || who_close(snaf[i]))
                continue;
            for(int ny = 2500; ny <=5000; ny += 1250)
            {
              //  cerr <<" AA" <<endl;
                int y = ny - snaf[i]->ny;
                int x =  16000- snaf[i]->nx;
                 //cerr << x << " " <<  y << endl;;
                double dist = sqrt((y * y) + (x * x));

                if (dist < dfmax_dist)
                {
                    for(int j = 0; j < 3; j++)
                    {

                        
                        if (whoisthere[j] == 0)
                        {   //   cerr <<whoisthere[j] <<"ccc"<< endl;
                            magic_cal(snaf[i], 16000, ny, dist);

                            return ;
                        }
                    }//cerr <<endl;
                }
            }
        }
    }
}
int main()
{
  
    int myTeamId; // if 0 you need to score on the right of the map, if 1 you need to score on the left
    cin >> myTeamId; cin.ignore();
    
    
    g_myTeamId = myTeamId;
    
    set_col();

   int mygoalx = 0;
   int opgoalx = 0;
   
   if (!myTeamId)
    opgoalx = 16000;
    else 
    mygoalx = 16000;
   t_node* st1 = new t_node;
    t_node* st2 = new t_node;
    memset(st1, 0, sizeof(t_node));
       memset(st2, 0, sizeof(t_node));
    st1->x = mygoalx;
    st1->y = 3750;
    st1->d_ux = 0.0;
    st1->d_uy = 0.0;
    st1->d_force = 0.0;
    st1->d_mass = 10000.0;
    st1->nx = st1->x;
    st1->ny = st1->y;
    st2->x = opgoalx;
    st2->y = 3750;
    st2->d_ux = 0.0;
    st2->d_uy = 0.0;
    st2->d_force = 0.0;
    st2->d_mass = 10000.0;
    st2->nx = st2->x;
    st2->ny = st2->y;
     st2->entityId = 100;
     st1->entityId = 200;
    mygoal_node = st1;
    enemygoal_node = st2;
    
    while (1) {    

        snaf.clear();
        wiz.clear();
        owiz.clear();
        bger.clear();
   
        int myScore;
        int myMagic;
        cin >> myScore >> myMagic; cin.ignore();
        int opponentScore;
        int opponentMagic;
        cin >> opponentScore >> opponentMagic; cin.ignore();
        int entities; // number of entities still in game
    
        cin >> entities; cin.ignore();
        for (int i = 0; i < entities; i++) {
            
            int entityId; // entity identifier
            string entityType; // "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" or "BLUDGER"
            int x; // position
            int y; // position
            int vx; // velocity
            int vy; // velocity
            int state; // 1 if the wizard is holding a Snaffle, 0 otherwise. 1 if the Snaffle is being held, 0 otherwise. id of the last victim of the bludger.
            cin >> entityId >> entityType >> x >> y >> vx >> vy >> state; cin.ignore();
            t_node* st = new t_node;
             memset(st, 0, sizeof(t_node));
            st->entityId = entityId;
            st->entityType = entityType;
            st->x = x;
            st->y = y;
            st->d_force = 0.0;
            st->d_uy = 0.0;
            st->d_ux = 0.0;
            st->vx = vx;
            st->check = 0;
            st->vy = vy;
            st->state = state;
            st->d_norm_vector = 0.0;
            st->direction = 0;
            memset(st->radar, 0, sizeof(st->radar));

        if (entityType == "SNAFFLE")
    { st->d_mass = MASS_SNAFFLES;
            snaf.push_back(st);
            st->d_force = 0.0;
           
    }
        else if (entityType == "WIZARD")
        {st->d_mass = MASS_WIZARD;
            wiz.push_back(st);
            st->d_force = 150.0;
        }
        else if (entityType == "BLUDGER")
        {st->d_mass = MASS_BLUDEGER;
            bger.push_back(st);
            st->d_force = 1000.0;
        }
        else if (entityType == "OPPONENT_WIZARD")
        {st->d_mass = MASS_WIZARD;
            owiz.push_back(st);
            st->d_force = 150.0;
        }
                    calculate_direction_and_vector(st);
            calculate_next(st);
        } 
         try{
        find_radar(wiz[0]);
        find_radar(wiz[1]);
         }     catch(int expn){
                cerr << "can't use" <<expn <<endl;
            }
            
        for (int i = 0; i < 2; i++) {
            use_magic = 0;
            possible_commend = 1;
            t_node *target;
                     cerr <<" B" <<endl;
                       try{
            target = what_is_close(wiz[i], wiz[!i]);
                       }
                 catch(int expn){
                cerr << "can't use" <<expn <<endl;
            }
          //    cerr <<" B" <<endl;
           // //enemygoal_node
            //mygoal_node
            try{
            defense();
            if (possible_commend)
                attack();
            if (possible_commend && wiz[i]->state == 1)
               shoot(wiz[i]);
            if (possible_commend)
               move(wiz[i], target);
            }
            catch(int expn){
                cerr << "can't use" <<expn <<endl;
            }
            cerr << i << endl;
           // cerr << target->can << endl;
           if(possible_commend)// if (wiz[0]->radar[0][2] == 3)
        {
                 cout << "MOVE " <<  target->x << " " << target->y << " 150" <<endl;
                 target->check = 1;
                 
        }
              //   cerr <<" A" <<endl;
      
        }   // cerr << snaf[i]->vx <<endl;
      cerr << bger[0]->nx <<" " << bger[0]->ny << endl;
        cerr << bger[0]->x <<" " << bger[0]->y << endl;
     cerr << wiz[0]->nx <<" " << wiz[0]->ny << endl;
        cerr << wiz[0]->direction <<endl;
      // cerr << calculate_direction(wiz[0], snaf[0])  << endl;
  for(int i = 0; i < 3; i++)
       {
           for (int j = 0;j < 3; j++)
           {if (wiz[0]->radar[i][j] != 0)
               cerr << wiz[0]->radar[i][j]->entityId<< " ";
               else
               cerr << 0 << " ";
           }cerr << endl;
       }cerr << endl;
     
        magic++;
    }
}