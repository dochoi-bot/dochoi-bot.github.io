#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <memory.h>
#include <utility>
#include <unordered_set>
#include <cmath>
#include <cstdlib>
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

int mygoalx = 0;
int opgoalx = 0;
int g_dx[9] = {0,1000,1000,0,-1000,-1000,-1000,0,1000};
int g_dy[9] = {0,0,1000,1000,1000,0,-1000,-1000,-1000};
int n_myScore = 0;
int n_oScoere = 0;
void calculate_next_bger_dt(double dt);
vector<pair <int, int >> random_pointv;
typedef struct s_node{
    
            int entityId = 0; // entity identifier
            string entityType = ""; // "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" or "BLUDGER"
            int x = 0; // position
            int y = 0; // position
            int vx = 0; // velocity
            int vy = 0; // velocity
			int nvx = 0;
			int nvy = 0;
            int state = 0;

            
            int direction = 0;
            double d_mass= 1.0;
            double d_radious = 1.0;
            double d_friction = 1.0;
            double d_force= 1.0;
            int nx= 0;
            int ny = 0;
            int tx = 0;
            int ty = 0;
            int check_collision = 0;
            int check = 0;
			vector<int> check_col_v;
			s_node* target = 0;
}t_node;

t_node* mygoal_node;
t_node* enemygoal_node;
vector<t_node*> snaf;
vector<t_node*> wiz;
vector<t_node*> owiz;
vector<t_node*> bger;
vector<t_node*> col;
vector<t_node*> next_nodes;
vector<t_node*> next_nodes2;
bool possible_commend;
bool use_magic;
int magic = 10;
double calculate_dist(t_node* node, t_node* node2)
{
    int y = node->y - node2->y;
    int x = node->x - node2->x;

    double dist = sqrt((y * y) + (x * x));
	if (dist < 50)
		return (0);
    return (dist);
}


void init_nodes()
{
	next_nodes.clear();
	for (int i = 0; i < 2; i++) {
		t_node* st = new t_node;
		memcpy(st, wiz[i], sizeof(t_node));
		next_nodes.push_back(st);
	}
	for (int i = 0; i < 2; i++) {
		t_node* st = new t_node;
		memcpy(st, owiz[i], sizeof(t_node));
		next_nodes.push_back(st);
	

	}
	for (int i = 0; i < 2; i++) {
		t_node* st = new t_node;
		memcpy(st, bger[i], sizeof(t_node));
		next_nodes.push_back(st);

	
	}
	for (int i = 0; i < snaf.size(); i++)
	{
		t_node* st = new t_node;
		memcpy(st, snaf[i], sizeof(t_node));
		next_nodes.push_back(st);

	}
}

void init_nodes2()
{
	next_nodes2.clear();
	for (int i = 0; i < next_nodes.size(); i++)
	{
		t_node* st = new t_node;
		memcpy(st, next_nodes[i], sizeof(t_node));
		next_nodes2.push_back(st);

	}
}
t_node *what_is_close(t_node *mywiz)
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

        return (target);
}


t_node *what_is_close_bger(t_node *node)
{
    if (snaf.size() == 1)
        return (snaf[0]);
    
    t_node *target = wiz[0];
    
    double min_dist = 9999999999.0;
    for (int i = 0; i < wiz.size(); i++)
    {
        if (node->state == wiz[i]->entityId)
            continue;
        double dist = calculate_dist(node, wiz[i]);
        if (dist < min_dist)
        {
            target = wiz[i];
            min_dist = dist;
        }
    }
    for (int i = 0; i < owiz.size(); i++)
    {
        if (node->state == owiz[i]->entityId)
            continue;
        double dist = calculate_dist(node, owiz[i]);
        if (dist < min_dist)
        {
            target = owiz[i];
            min_dist = dist;
        }
    }

        return (target);
}



pair<double , double> to_unit_vector(int x, int y)
{
    pair <double ,double > uv;
    
     double norm = sqrt((x * x) + (y * y));

    uv.first = (double)(y) / norm;
    uv.second = (double)(x) / norm;
         if (norm < 100.0)
     {
     uv.first = 0;
     uv.second = 0;
     }
    return (uv);
}

void calculate_next_dt(t_node *node, double dt)
{
	node->x = node->x + (int)round(dt * (node->vx));
	node->y = node->y + (int)round(dt * (node->vy));

}

void calculate_nextv(t_node* node, int tx, int ty)
{

	pair<double, double> d_uv;
	int dy = ty - node->y;
	int dx = tx - node->x;
	d_uv = to_unit_vector(dx, dy);
	double d_uy = d_uv.first;
	double d_ux = d_uv.second;
	if (node->vx == 0 && node->vy == 0)
	{
		node->y = node->y;
		node->x = node->x;
		node->direction = -1;
		return;
	}

	node->vx = (int)round(((double)(node->vx) + (d_ux * (node->d_force / node->d_mass))));
	node->vy = (int)round(((double)(node->vy) + (d_uy * (node->d_force / node->d_mass))));
}

void caculateadd()
{
	for (int i = 0; i < next_nodes.size(); i++)
	{
		if (next_nodes[i]->target != 0)
			calculate_nextv(next_nodes[i], next_nodes[i]->target->x, next_nodes[i]->target->y );
	}
}

void caculatev2p(double dt)
{
	for (int i = 0; i < next_nodes.size(); i++)
	{
		calculate_next_dt(next_nodes[i], dt);
	}
}
void is_collision(t_node* node, t_node* target, double dt)
{
	for (int i = 0; i < node->check_col_v.size(); i++)
	{
		if (node->check_col_v[i] == target->entityId)
			return;
	}
	for (int i = 0; i < target->check_col_v.size(); i++)
	{
		if (target->check_col_v[i] == node->entityId)
			return;
	} 
	if (target->entityType == "SNAFFLE" && target->state == 1)
		return;
	if (node->entityType == "WIZARD" && target->entityType == "SNAFFLE")
	{
		double dist = calculate_dist(node, target);
		if (dist <= node->d_radious - 1)
		{
			//cerr << "AAA" << endl;
		target->x = node->x;
		target->y = node->y;
		target->vx = node->vx;
		target->vy = node->vy;
		target->state = 1;
		node->state = 1;
		}
		return;
	}
	else if (node->entityType == "OPPONENT_WIZARD" && target->entityType == "SNAFFLE")
	{
		double dist = calculate_dist(node, target);
		if (dist <= node->d_radious - 1)
		{

		target->x = node->x;
		target->y = node->y;
		target->vx = node->vx;
		target->vy = node->vy;
		target->state = 1;
		node->state = 1;
	}
		return;
	}
				//cerr << "GOGO" << endl;

				double mcoeff = (node->d_mass + target->d_mass) / (node->d_mass * target->d_mass);
				double nx = (double)(node->x - target->x);
				double ny = (double)(node->y - target->y);
				double nxnydeux = (nx * nx) + (ny * ny);
				double dvx = (double)(node->vx - target->vx);
				double dvy = (double)(node->vy - target->vy);
				double product = ((nx * dvx) + (ny * dvy)) / (nxnydeux * mcoeff);
				double fx = nx * product;
				double fy = ny * product;
				double m1c = 1.0 / node->d_mass;
				double m2c = 1.0 / node->d_mass;

				node->nvx -= fx * m1c;
				node->nvy -= fy * m1c;

				target->nvx += fx * m2c;
				target->nvy += fy * m2c;
				double impulse = sqrt((fx * fx) + (fy * fy));
				if (impulse < 100.0) {
					double min = 100.0 / impulse;
					fx = fx * min;
					fy = fy * min;
				}

				node->vx -= fx * m1c;
				node->vy -= fy * m1c;
				target->vx += fx * m2c;
				target->vy += fy * m2c;

	}

void col_collision(t_node* node, t_node* target)
{

	double mcoeff = (node->d_mass + target->d_mass) / (node->d_mass * target->d_mass);
	double nx = (double)(node->x - target->x);
	double ny = (double)(node->y - target->y);
	double nxnydeux = (nx * nx) + (ny * ny);
	double dvx = (double)(node->vx - target->vx);
	double dvy = (double)(node->vy - target->vy);
	double product = ((nx * dvx) + (ny * dvy)) / (nxnydeux * mcoeff);
	double fx = nx * product;
	double fy = ny * product;
	double m1c = 1.0 / node->d_mass;
	double m2c = 1.0 / node->d_mass;

	node->nvx -= fx * m1c;
	node->nvy -= fy * m1c;

	node->vx -= fx * m1c;
	node->vy -= fy * m1c;


}

void wall_collistion(t_node* node)
{
	if (node->x < 0)
	{
		node->x = 0;
		node->vx *= -1;
	}
	if (node->x > 16000)
	{
		node->x = 16000;
		node->vx *= -1;
	}
	if (node->y < 0)
	{
		node->y = 0;
		node->vy *= -1;
	}
	if (node->y > 7500)
	{
		node->y = 7500;
		node->vy *= -1;
	}
}
void addthorw()
{
	for (int i = 0; i < snaf.size();i++)
	{
		if (snaf[i]->state == 1)
		{
			for (int j = 0;j < wiz.size(); j++)
			{
				if (snaf[i]->x == wiz[j]->x && snaf[i]->y == wiz[j]->y)
				{
					snaf[i]->vx = wiz[j]->vx;
					snaf[i]->vy = wiz[j]->vy;
					snaf[i]->d_force = 500;
					snaf[i]->target = wiz[j]->target;
				}
			}
			for (int j = 0;j < owiz.size(); j++)
			{
				if (snaf[i]->x == owiz[j]->x && snaf[i]->y == owiz[j]->y)
				{
					snaf[i]->vx = owiz[j]->vx;
					snaf[i]->vy = owiz[j]->vy;
					snaf[i]->d_force = 500;
					snaf[i]->target = mygoal_node;
				}
			}
		}
	}
}
void next_turn()
{
	double dt = 0.2;
	caculateadd();
	for (int time = 0; time  < 5; time++)
	{
		
		for (int i = 0; i < next_nodes.size(); i++)
		{
		
			bool flag = 1;
			for (int j = 0; j < next_nodes.size(); j++)
			{
				
				if (next_nodes[i]->entityId == next_nodes[j]->entityId)
					continue;

				double dist = calculate_dist(next_nodes[i], next_nodes[j]);
				if (dist < next_nodes[i]->d_radious + next_nodes[j]->d_radious)
				{
				
					is_collision(next_nodes[i], next_nodes[j], dt);
					next_nodes[i]->check_col_v.push_back(next_nodes[j]->entityId);
					next_nodes[j]->check_col_v.push_back(next_nodes[i]->entityId);
					flag = 0;
				}
			}
	
			for (int j = 0; j < col.size(); j++)
			{
				double dist = calculate_dist(next_nodes[i], col[j]);
				if (dist < next_nodes[i]->d_radious + col[j]->d_radious)
				{
					col_collision(next_nodes[i], col[j]);
					flag = 0;
				}
			}
			wall_collistion(next_nodes[i]);
			
		}
		caculatev2p(dt);
		//cerr << "cc" << endl;
	}
}

double weight()
{
	double value = 0;
	for (int i = 0; i < next_nodes.size(); i++)
	{
		if (next_nodes[i]->entityType == "SNAFFLE" || next_nodes[i]->entityType =="WIZARD")
		{
			if (next_nodes[i]->entityType == "SNAFFLE")
			{
				for (int j = 0; j < next_nodes.size(); j++)
				{
					if (next_nodes[j]->entityType == "WIZARD")
					{
						double dist = calculate_dist(next_nodes[i], next_nodes[j]);

						value +=dist;
					}
				}
						value += calculate_dist(next_nodes[i], enemygoal_node);
			}
			else
				value += calculate_dist(next_nodes[i], enemygoal_node);
		}
	}

	return value;
}

void set_fric()
{// 벡터초기화
	for (int i = 0; i < next_nodes.size(); i++)
	{
		next_nodes[i]->vx = round((int)((double)(next_nodes[i]->vx) * next_nodes[i]->d_friction));
		next_nodes[i]->vy = round((int)((double)(next_nodes[i]->vy) * next_nodes[i]->d_friction));
		next_nodes[i]->check_col_v.clear();
	}

}
void calculate_next_snaf()
{
	for (int i = 0; i < snaf.size();i++)
	{
		if (snaf[i]->state == 1)
		{
			for (int j = 0;j < wiz.size(); j++)
			{
				if (snaf[i]->x == wiz[j]->x && snaf[i]->y == wiz[j]->y)
				{
					snaf[i]->nx = wiz[j]->nx;
					snaf[i]->ny = wiz[j]->ny;

					snaf[i]->nvx = wiz[j]->nvx;
					snaf[i]->nvy = wiz[j]->nvy;
				}
			}
			for (int j = 0;j < owiz.size(); j++)
			{
				if (snaf[i]->x == owiz[j]->x && snaf[i]->y == owiz[j]->y)
				{
					snaf[i]->nx = owiz[j]->nx;
					snaf[i]->ny = owiz[j]->ny;

					snaf[i]->nvx = owiz[j]->nvx;
					snaf[i]->nvy = owiz[j]->nvy;
				}
			}
		}
		else
		{
			snaf[i]->nx = snaf[i]->x + snaf[i]->vx;

			snaf[i]->ny = snaf[i]->y + snaf[i]->vy;
			snaf[i]->nvx = snaf[i]->vx;
			snaf[i]->nvy = snaf[i]->vy;
		}
	}
}

void calculate_next(t_node *node, int tx, int ty)
{

    pair<double , double> d_uv;
    int dy = ty - node->y;  
    int dx = tx - node->x;
    d_uv = to_unit_vector(dx, dy);
    double d_uy = d_uv.first;
    double d_ux = d_uv.second;
    if (node->vx == 0 && node->vy == 0)
        {
            node->ny = node->y;
            node->nx = node->x;
            node->direction = -1;
            return ;
        }
    node->nx = node->x  + (node->vx) + (int)round(d_ux * (node->d_force / node->d_mass));
    node->ny = node->y  + (node->vy) + (int)round(d_uy * (node->d_force / node->d_mass));
	node->nvx = (node->vx) + (int)round(d_ux * (node->d_force / node->d_mass));
	node->nvy = (node->vy) + (int)round(d_uy * (node->d_force / node->d_mass));

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
	st3->x = 0;
	st3->y = 5750;
	st3->d_mass = 10000;
	st4->x = 0;
	st4->d_mass = 10000;
	st4->y = 1750;
	st5->x = 16000;
	st5->d_mass = 10000;
	st5->y = 5750;
	st6->x = 16000;
	st6->y = 1750;
	st6->d_mass = 10000;

	st3->entityId = -10;
	st4->entityId = -10;
	st5->entityId = -10;
	st6->entityId = -10;

	st3->entityType == "COL";
	st4->entityType == "COL";
	st5->entityType == "COL";
	st6->entityType == "COL";
	
	col.push_back(st3);
	col.push_back(st4);
	col.push_back(st5);
	col.push_back(st6);
	for (int i = 0; i < 4; i++)
	{
		col[i]->d_radious = 300.0;
	}
}


void init_node(t_node* st, string entityType)
{
      if (entityType == "SNAFFLE")
    { st->d_mass = MASS_SNAFFLES;
        st->d_force = 0.0;
        st->d_radious = RADIUS_SNAFFLES;
        st->d_friction = FRICTION_SNAFFLES;
            snaf.push_back(st);      
    }
        else if (entityType == "WIZARD")
        {st->d_mass = MASS_WIZARD;
        st->d_friction = FRICTION_WIZARD;
         st->d_radious = RADIUS_WIZARD;
            st->d_force = 150.0;
                    wiz.push_back(st);
        }
        else if (entityType == "BLUDGER")
        {st->d_mass = MASS_BLUDEGER;
         st->d_friction = FRICTION_BLUDEGER;
         st->d_radious = RADIUS_BLUDEGER;
            bger.push_back(st);
            st->d_force = 1000.0;
        }
        else if (entityType == "OPPONENT_WIZARD")
        {            st->d_mass = MASS_WIZARD;
                     st->d_friction = FRICTION_WIZARD;
                     st->d_force = 150.0;
                      st->d_radious = RADIUS_WIZARD;
            owiz.push_back(st);

        }
        //            calculate_direction_and_vector(st);
          
      
}


int main()
{
  
    int myTeamId; // if 0 you need to score on the right of the map, if 1 you need to score on the left
    cin >> myTeamId; cin.ignore();
    
    
	set_col();

	for (int i = 0; i <= 16000; i+=8000)
	{
		random_pointv.push_back(make_pair(i, 0));
	}
	for (int i = 0; i <= 16000; i += 8000)
	{
		random_pointv.push_back(make_pair(i, 7500));
	}
	random_pointv.push_back(make_pair(16000, 3750));
	//random_pointv.push_back(make_pair(i, 7500));
	//for (int i = 5000; i <= 7500; i += 2500)
	//{
	//	random_pointv.push_back(make_pair(0, i));
	//}
	/*for (int i = 0; i <= 7500; i += 2500)
	{
		random_pointv.push_back(make_pair(16000, i));
	}*/
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

   st1->nx = st1->x;
   st1->ny = st1->y;
   st2->x = opgoalx;
   st2->y = 3750;

   st2->d_force = 0.0;
   st2->d_mass = 10000.0;

   st2->entityId = 100;
   st1->entityId = 200;
   mygoal_node = st1;
   enemygoal_node = st2;

   while (1) {

	   snaf.clear();
	   wiz.clear();
	   owiz.clear();
	   bger.clear();
	   next_nodes2.clear();
	   next_nodes.clear();
	   int myScore;
	   int myMagic;
	   cin >> myScore >> myMagic; cin.ignore();
	   int opponentScore;
	   int opponentMagic;
	   cin >> opponentScore >> opponentMagic; cin.ignore();
	   int entities; // number of entities still in game
	   n_myScore = myScore;
	   n_oScoere = opponentScore;
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
		   st->vx = vx;
		   st->check_collision = 0;
		   st->vy = vy;
		   st->state = state;
		   init_node(st, entityType);
	   }
	   t_node* target;
	   t_node* target_o;
	   t_node* target_b;
	   for (int i = 0; i < 2; i++) {
		   target = what_is_close(wiz[i]);
		   target_b = what_is_close_bger(bger[i]);
		   target_o = what_is_close(owiz[i]);
		  
		   owiz[i]->target = target_o;
		   bger[i]->target = target_b;
	   }
	   init_nodes();
	   t_node* target1 = new t_node;
	   t_node* target2 = new t_node;
	   memset(target1, 0, sizeof(t_node));
	   memset(target2, 0, sizeof(t_node));
	   wiz[0]->target = target1;
	   wiz[1]->target = target2;
	   pair<int, int > best[2];
	   vector<t_node*> temp;
	   double standard = 9999999999990.0;

	   for (int i = 0; i < random_pointv.size(); i++)
	   {
		   for (int j = 0; j < random_pointv.size(); j++)
		   {
			   init_nodes();
			   target1->x = random_pointv[i].first;
			   target1->y = random_pointv[i].second;
			   target2->x = random_pointv[j].first;
			   target2->y = random_pointv[j].second;
			   next_turn();
			   set_fric();
			   temp = next_nodes;
			   init_nodes2();
				   for (int a = 0; a < random_pointv.size(); a++)
				   {
					   for (int b = 0; b < random_pointv.size(); b++)
					   {
						  
						   next_nodes = next_nodes2;
						   
						   target1->x = random_pointv[a].first;
						   target1->y = random_pointv[a].second;
						   target2->x = random_pointv[b].first;
						   target2->y = random_pointv[b].second;
						   next_turn();
						  
						   double  dist = weight();
						   if (standard > dist)
						   {
							   best[0] = random_pointv[i];
							   best[1] = random_pointv[j];
							   standard = dist;

						   }
						   next_nodes = temp;
					   }
				   }
				  
		   }
	   }

	   init_nodes();
	   next_turn();
	   set_fric();
			for (int i = 0; i < 2; i++) {
				use_magic = 0;
				possible_commend = 1;
				target = what_is_close(wiz[i]);
			
				if(wiz[i]->state == 0)
					cout << "MOVE " << best[i].first << " " << best[i].second << " 150" << endl;
				else
					cout << "THROW " << best[i].first << " " << best[i].second << " 500" << endl;
			}
		/*	 for (int i = 0; i < next_nodes.size(); i++)
			 {
	
					 cerr << next_nodes[i]->entityType << " " << next_nodes[i]->entityId << "예측 위치 :" << next_nodes[i]->x << " " << next_nodes[i]->y << endl;
			 }*/
     cerr<< rand() << endl;
        magic++;
    }
}