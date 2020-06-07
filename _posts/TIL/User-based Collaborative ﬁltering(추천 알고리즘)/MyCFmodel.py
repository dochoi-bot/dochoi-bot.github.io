# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    MyCFmodel.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: dochoi <dochoi@student.42seoul.kr>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/01 16:42:40 by dochoi            #+#    #+#              #
#    Updated: 2020/06/01 16:42:40 by dochoi           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import math

def simil(x,y): # 식 2
    xdotv = 0.0 # 분자, x 와 v 의 dot product
    sum_powx =0.0 # x의 원소의 제곱의 합
    sum_powy = 0.0 # y의 원소의 제곱의 합
    for key in range(1 ,num_items+ 1) :
        if key in y and key in x: # x, y의 key아이템 평가가 동시에 존재하면
            xdotv += x[key] * y[key]
            sum_powx += x[key] * x[key]
            sum_powy += y[key] * y[key]
        elif key in x: # x에만 key평가가 존재하면
            sum_powx += x[key] * x[key]
        elif key in y :# y에만 key평가가 존재하면
            sum_powy += y[key] * y[key]
    sqrt_sum_powx_powy = math.sqrt(sum_powx) * math.sqrt(sum_powy)
    if math.isclose(sqrt_sum_powx_powy , 0.0): # 0으로 나누는 예외처리, 실수 값 비교연산 주의
        return 0
    return xdotv / sqrt_sum_powx_powy

def mean_rating(user):
    sum_rating  = 0.0
    cnt = 0
    for key in rating_table[user].keys(): # user user_mean_rating 구하기
        sum_rating += rating_table[user][key]
        cnt += 1
    if cnt == 0: #0으로 나누는 예외처리
        return 0
    return sum_rating / cnt


def target_user_rating_predict(t_user,U_prime): # 식 1

    target_user_mean_rating = mean_rating(t_user) # t_user의 rating 평균값
    len_t_user_rating = 0 # t_user의 rating 개수

    k = {} #k를 item마다 구해줘야 하기 때문에 dict를 이용하였다.
    reco_dict = {} #추천 item을 담을 dict

    for i in range(1 ,num_items+ 1) : #추천용 dict 생성(target유저가 평가하지 않고 다른유저가 평가한 아이템 목록 생성)
        for other in U_prime: #  other[simil, 유저번호]
            if i not in rating_table[t_user] and i in rating_table[other[1]]: # targt 유저가 rating하지 않고, 다른유저가 rating한 item만 본다.
                    reco_dict[i] = 0.0
                    k[i] = 0.0

    for i in reco_dict.keys(): #추천 목록 후보들
        for other in U_prime: #  other[simil, 유저번호]
            if i in  rating_table[other[1]]:  #itme을 other users가 평가했으면
                if i in reco_dict:
                    k[i] +=abs(other[0]) #k값 구하기
                    reco_dict[i] +=(other[0] * (rating_table[other[1]][i] - mean_rating(other[1]))) # 1식의 시그마 안쪽 수식

    for key in reco_dict.keys():
        k_i = 0.0
        if not math.isclose(k[key], 0.0): #k값 0으로 나누는 에러 처리 (float 비교연산 주의 )
            k_i = 1 / k[key]
        reco_dict[key] = target_user_mean_rating  + ((k_i) * reco_dict[key]) #1식의 시그마 바깥쪽 수식
    return reco_dict

rating_table = [{}] ## 레이팅 테이블

sys.stdin = open("input0.txt", 'r')
num_sim_user_topk = int(input()) # 유사한 유저의 수
num_item_rec_topk = int(input()) # 추천할 아이템의 개수
num_users = int(input()) # 총 유저의 수
for i in range(1, num_users + 1):
    rating_table.append({})
num_items = int(input()) # 총 아이템의 수
num_rows = int(input()) # 총 rating 개수

for i in (range(int(num_rows))):
    temp = input().split()
    rating_table[int(temp[0])][int(temp[1])] = float(temp[2]) # rating_talbe[유저번호][아이템번호][rating값]
num_reco_users = int(input()) # 추천 유저의 수
for i in (range(int(num_reco_users))):
    num_reco_user = int(input())
    U_prime = [] # 최근접 이웃 집합
    for j in range(1, num_users + 1):
        if num_reco_user != j:
            U_prime.append([simil(rating_table[num_reco_user], rating_table[j]), j]) # [simil, 유저번호] appned
    U_prime.sort() # sort
    if len(U_prime) > num_sim_user_topk: # num_sim_user_topk개수만큼 잘라준다.
        del U_prime[:len(U_prime) - num_sim_user_topk]
    reco_dict_ = target_user_rating_predict(num_reco_user ,U_prime)
    reco_dict_ = sorted(reco_dict_, key=lambda k : reco_dict_[k], reverse=True) # sort
    cnt = 0
    for key in reco_dict_:  # 상위 n개 출력하기
        print(key,end=' ')
        cnt += 1
        if cnt == num_item_rec_topk:
            break
    print()

