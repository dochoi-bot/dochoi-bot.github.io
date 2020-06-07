#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <memory.h>
using namespace std;

/**
 * Grab Snaffles and try to throw them through the opponent's goal!
 * Move towards a Snaffle to grab it and use your team id to determine towards where you need to throw it.
 * Use the Wingardium spell to move things around at your leisure, the more magic you put it, the further they'll move.
 **/
 

typedef struct node{
            int entityId; // entity identifier
            string entityType; // "WIZARD", "OPPONENT_WIZARD" or "SNAFFLE" or "BLUDGER"
            int x; // position
            int y; // position
            int vx; // velocity
            int vy; // velocity
            int state;
}node;

    

int goals_x[] = {16000, 0};
int main()
{
    int myTeamId; // if 0 you need to score on the right of the map, if 1 you need to score on the left
    cin >> myTeamId; cin.ignore();
int magic = 10;
    // game loop
    while (1) {
        
        int myScore;
        int myMagic;
        cin >> myScore >> myMagic; cin.ignore();
        int opponentScore;
        int opponentMagic;
        cin >> opponentScore >> opponentMagic; cin.ignore();
        int entities; // number of entities still in game
        vector<node> snaf;
        vector<node> wiz;
        vector<node> owiz;
        vector<node> bger;
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
            node st;
            st.entityId = entityId;
            st.entityType = entityType;
            st.x = x;
            st.y = y;
            st.vx = vx;
            st.vy = vy;
            st.state = state;
        if (entityType == "SNAFFLE")
        {  
            snaf.push_back(st);
        }
        else if (entityType == "WIZARD")
        {
            wiz.push_back(st);
        
        }
        else if (entityType == "BLUDGER")
        {
            bger.push_back(st);
        
        }
        else if (entityType == "OPPONENT_WIZARD")
        {
            owiz.push_back(st);
        
        }
        }   
        
  
        
        
        
        
        

            int flag2 = 0;
        int tx = goals_x[myTeamId];
            int ty = 3750;
            int myx = goals_x[!myTeamId];
              int prex = -1;
            int prey = - 1;
            
            int myroom = opponentScore;
            int obroom = myScore;
              for(int j = 0 ;j < snaf.size(); j++)
                {
                    if (myTeamId == 0)
                    {
                        if (snaf[j].x < 8000)
                            myroom++;
                        else
                            obroom++;
                    }
                    else if (myTeamId == 1)
                    {
                         if (snaf[j].x > 8000)
                            myroom++;
                        else
                            obroom++;
                    }
                }
                if (myroom > obroom)
                {
                    flag2 = 1;
               
                }
                else if (myroom < obroom)
                    flag2 = 2;
            bool flag4 = 1;        
                for(int j = 0 ; j < owiz.size(); j++)
                {
                      if (owiz[j].x < 7800 && myTeamId == 0)
                      {
                          flag4 = 0;
                          break;
                      }
                      else if (owiz[j].x > 8200 && myTeamId == 1)
                        {
                          flag4 = 0;
                          break;
                      }
                }
                if (flag4)
                    flag2 = 0;
        for (int i = 0; i < 2; i++) {
              int bst_snafx;
            int bst_snafy;
            int bst_snafyidx;
            int bst_snafyidx_m;
            int bst_snafx_m;
            int bst_snafy_m;
          
           int close = 0;
           int closex;
            int closey;
            double mindst = 999999999999999999.0;
            double mindst2 = 999999999999999999.0;

            for(int j = 0 ;j < snaf.size(); j++)
                {    
                    
                    double dist = pow(wiz[i].x - snaf[j].x, 2) + pow(wiz[i].y - snaf[j].y, 2);
                       double dist2 = pow(myx - snaf[j].x, 2) + pow(3750 - snaf[j].y,2);
                              if (dist2 < mindst2)
                        {
                            bst_snafyidx_m = snaf[j].entityId;
                            bst_snafx_m = snaf[j].x;
                         bst_snafy_m = snaf[j].y;
                         mindst2  = dist2;
                        }

                        if (dist < mindst && (snaf.size() == 1 || (snaf[j].x != prex && snaf[j].y != prey)))
                        {
                            bst_snafyidx = snaf[j].entityId;
                            bst_snafx = snaf[j].x;
                         bst_snafy = snaf[j].y;
                         mindst  = dist;
                         if (dist < 910000)
                         {
                            close = 1;
                            closex = bst_snafx;
                            closey = bst_snafy;
                         }
                            
                        }
                }
             bool flag = 1;
             if (magic >= 60 && myScore <= opponentScore && close == 0)
            {
               
              for(int j = 0 ; j < snaf.size(); j++)
                {
                      
                          double dist = pow(myx - snaf[j].x, 2) + pow(ty - snaf[j].y, 2);
                     
                         if (dist < 15000000.0 && snaf[j].state == 0)
                         {
                             cout << "WINGARDIUM " << snaf[j].entityId << " " << tx <<" " << ty << " 60" << endl;
                            magic -= 60;
                         flag = 0;
                             break;
                    
                         }
                }
              
            }
            if (magic >= 30 && flag2 != 2 && close == 0)
            {
               
              for(int j = 0 ; j < snaf.size(); j++)
                {
                      
                          double dist = pow(tx - snaf[j].x, 2) + pow(ty - snaf[j].y, 2);
                     
                         if (dist < 12000000)
                         {
                             cout << "WINGARDIUM " << snaf[j].entityId << " " << tx <<" " << ty << " 30" << endl;
                            magic -= 30;
                         flag = 0;
                             break;
                         }
                         
                         else if (dist < 20000000 && magic >= 40)
                         {
                             cout << "WINGARDIUM " << snaf[j].entityId << " " << tx <<" " << ty << " 40" << endl;
                            magic -= 40;
                         flag = 0;
                             break;
                         }
                          else if (dist < 40000000 && magic >= 50)
                         {
                             cout << "WINGARDIUM " << snaf[j].entityId << " " << tx <<" " << ty << " 50" << endl;
                            magic -= 50;
                         flag = 0;
                             break;
                         }
                
                
                
                }
              
            }
            if (flag2 == 1)
            {
                 mindst = 999999999999999999.0;
           mindst2 = 999999999999999999.0;
           if (1)
           {
                for(int j = 0 ;j < snaf.size(); j++)
                {
                  
                        double dist = pow(wiz[i].x - snaf[j].x, 2) + pow(wiz[i].y - snaf[j].y, 2);
                        if (snaf[j].x < 8000 && myTeamId == 0)
                        {
                             if (dist < mindst && (snaf.size() == 1 || (snaf[j].x != prex && snaf[j].y != prey)))
                            {
                            bst_snafyidx = snaf[j].entityId;
                            bst_snafx = snaf[j].x;
                         bst_snafy = snaf[j].y;
                         mindst  = dist;
                        }
                        }
                        else if (snaf[j].x > 8000 && myTeamId == 1)
                        {
                        if (dist < mindst && (snaf.size() == 1 || (snaf[j].x != prex && snaf[j].y != prey)))
                        {
                            bst_snafyidx = snaf[j].entityId;
                            bst_snafx = snaf[j].x;
                         bst_snafy = snaf[j].y;
                         mindst  = dist;
                        }
                        }
                }
                  if (flag && magic >= 60 && close == 0)
                {
                    cout << "WINGARDIUM " << bst_snafyidx << " " << tx <<" " << ty << " 60" << endl;
                        magic -= 60;
                        flag = 0;
                }
           }
            }
           else if (flag2 == 2)
            {
                 mindst = 999999999999999999.0;
           mindst2 = 999999999999999999.0;

           if (1)
           {
       
                for(int j = 0 ;j < snaf.size(); j++)
                {
                  
                        double dist = pow(wiz[i].x - snaf[j].x, 2) + pow(wiz[i].y - snaf[j].y, 2);
                        double dist3 = pow(myx - snaf[j].x, 2) + pow(3750- snaf[j].y, 2);
                        if (snaf[j].x > 7800 && myTeamId == 0)
                        {
                           
                             if (dist < mindst && (snaf.size() == 1 || (snaf[j].x != prex && snaf[j].y != prey)))
                            {
                            bst_snafyidx = snaf[j].entityId;
                            bst_snafx = snaf[j].x;
                         bst_snafy = snaf[j].y;
                         mindst  = dist;
                         
                            }
                        }
                        else if (snaf[j].x < 8200 && myTeamId == 1)
                        {
                        if (dist < mindst && (snaf.size() == 1 || (snaf[j].x != prex && snaf[j].y != prey)))
                        { 
                            bst_snafyidx = snaf[j].entityId;
                            bst_snafx = snaf[j].x;
                         bst_snafy = snaf[j].y;
                         mindst  = dist;
                       
                        }
                        }
                }
                
              
           }
            }
               if (flag && magic >= 60)
                {
                    cout << "WINGARDIUM " << bst_snafyidx_m  << " " << tx <<" " << ty << " 60" << endl;
                        magic -= 60;
                        flag = 0;
                }
             if (wiz[i].state == 1 && flag)
            {
                for(int j = 0; j < bger.size();j++)
                {
                     double dist = pow(wiz[i].x - bger[j].x, 2) + pow(wiz[i].y - bger[j].y, 2);
                     double dist2 = pow(tx - bger[j].x, 2) + pow(ty - bger[j].y, 2);
                      double dist3 = pow(tx - wiz[i].x, 2) + pow(ty - wiz[i].y, 2);
                    
                    if (dist < 1000000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << tx << " " << ty << " 50" << endl;
                        flag = 0;
                   
                        break;
                    }
                    else if (dist < 810000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << tx << " " << ty << " 40" << endl;
                        flag = 0;
                   
                        break;
                    }
                    else if (dist < 700000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << tx << " " << ty << " 30" << endl;
                        flag = 0;
               
                        break;
                    }
                    else if (dist < 500000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << tx << " " << ty << " 20" << endl;
                        flag = 0;
                 
                        break;
                    }
                }
                if (flag)
                {
                    cout << "THROW " << tx << " " << ty << " 500" << endl;
                    close = 2;

                }
            }
            else if (flag && wiz[i].state == 0)
            {
                if (close == 1)
                {bst_snafx = closex;
                bst_snafy= closey;
                }
           for(int j = 0; j < bger.size();j++)
                {
                     double dist = pow(wiz[i].x - bger[j].x, 2) + pow(wiz[i].y - bger[j].y, 2);
                     double dist2 = pow(tx - bger[j].x, 2) + pow(ty - bger[j].y, 2);
                      double dist3 = pow(tx - wiz[i].x, 2) + pow(ty - wiz[i].y, 2);
                      if(dist < 4000000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 130" << endl;
                        flag = 0;

                        break;
                    }
                    else if(dist < 2000000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 100" << endl;
                        flag = 0;

                        break;
                    }
                    else if(dist < 810000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 80" << endl;
                        flag = 0;

                        break;
                    }
                    else if (dist < 700000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 70" << endl;

                        break;
                    }
                    else if (dist < 500000.0 && dist2 < dist3)
                    {
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 40" << endl;
                        flag = 0;

                        break;
                    }
                }
                if (flag)
                {                    
                    double dist = pow(wiz[i].x - bst_snafx, 2) + pow(bst_snafy - wiz[i].y, 2);
                    if(dist < 4000000.0)
                    {
                       cout << "MOVE " << bst_snafx << " " << bst_snafy << " 140" << endl;
                    }
                    else if(dist < 2000000.0)
                    {
                       cout << "MOVE " << bst_snafx << " " << bst_snafy << " 120" << endl;
                    }
                    else if(dist < 1400000.0 )
                    {
                       cout << "MOVE " << bst_snafx << " " << bst_snafy << " 100" << endl;
                    }
                    else if(dist < 900000.0 )
                    {
                       cout << "MOVE " << bst_snafx << " " << bst_snafy << " 80" << endl;
                    }
                    else if (dist < 700000.0 )
                    {
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 60" << endl;
                    }
                    else if (dist < 500000.0 )
                    {
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 50" << endl;
                    }
                    else
                        cout << "MOVE " << bst_snafx << " " << bst_snafy << " 150" << endl;

                }                    
            }
                  if (close != 2 )
            {
                prex = bst_snafx;
                prey = bst_snafy;
            }
            }
      
            // Write an action using cout. DON'T FORGET THE "<< endl"
            // To debug: cerr << "Debug messages..." << endl;
            // Edit this line to indicate the action for each wizard (0 ≤ thrust ≤ 150, 0 ≤ power ≤ 500, 0 ≤ magic ≤ 1500)
            // i.e.: "MOVE x y thrust" or "THROW x y power" or "WINGARDIUM id x y magic"
  magic++;
        }       
        
      
    }