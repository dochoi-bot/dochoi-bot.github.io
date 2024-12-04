---
layout: post
title: "IBM Cloud로 자연어 분석하기"
date: 2020-07-31
category : cloud
excerpt: "IBM Cloud로 자연어 분석하기"
tags: [Cloud]
comments: true
---







# IBM Cloud로 자연어 분석하기



IBM클라우드로 웹페이지의 자연어를 분석해보자.

<img src="https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic38.png" alt="pic38" style="zoom:25%;" />

왓슨으로 들어가준다.



<img src="https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic39.png" alt="pic39" style="zoom:33%;" />

<img src="https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic40.png" alt="pic40" style="zoom:50%;" />

자연어 이해하기 항목을 찾을 수 있다.



<img src="https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic41.png" alt="pic41" style="zoom:33%;" />

Lite플랜을 고르고 작성한다. 무료이기 때문이다.





![pic34](https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic34.png)

![pic35](https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic35.png)

![pic36](https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic36.png)



간단한 튜토리얼을 공부하였다.



![pic42](https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/pic42.png)

감지가능언어에 한국어가 있어서 한국어도 분석해 볼 수 있다.

![image-20200731230429406](https://raw.githubusercontent.com/dochoi-bot/dochoi-bot.github.io/master/_posts/Cloud/images/image-20200731230429406.png)

자신의 API 키와 URL을 확인하여 클라우드에 요청을 보낸다.



http://newsroom.ibm.com/Guerbet-and-IBM-Watson-Health-Announce-Strategic-Partnership-for-Artificial-Intelligence-in-Medical-Imaging-Liver",  사이트 분석이다

```shell
{
  "usage": {
    "text_units": 1,
    "text_characters": 4483,
    "features": 5
  },
  "sentiment": {
    "document": {
      "score": 0.695162,
      "label": "positive"
    }
  },
  "retrieved_url": "https://newsroom.ibm.com/Guerbet-and-IBM-Watson-Health-Announce-Strategic-Partnership-for-Artificial-Intelligence-in-Medical-Imaging-Liver",
  "language": "en",
  "keywords": [
    {
      "text": "liver cancer",
      "relevance": 0.724552,
      "count": 3
    },
    {
      "text": "Watson Health",
      "relevance": 0.711929,
      "count": 3
    },
    {
      "text": "IBM Watson Health plan",
      "relevance": 0.615298,
      "count": 1
    },
    {
      "text": "interventional medical imaging",
      "relevance": 0.611459,
      "count": 1
    },
    {
      "text": "artificial intelligence",
      "relevance": 0.607823,
      "count": 3
    },
    {
      "text": "liver cancer diagnostics",
      "relevance": 0.604653,
      "count": 1
    },
    {
      "text": "market software solutions",
      "relevance": 0.58,
      "count": 1
    },
    {
      "text": "Watson Imaging Care Advisor",
      "relevance": 0.578128,
      "count": 2
    },
    {
      "text": "medical imaging",
      "relevance": 0.553427,
      "count": 2
    },
    {
      "text": "important field",
      "relevance": 0.550298,
      "count": 1
    },
    {
      "text": "IBM Watson Health",
      "relevance": 0.543361,
      "count": 2
    },
    {
      "text": "IBM",
      "relevance": 0.54311,
      "count": 2
    },
    {
      "text": "clinical decision support solutions",
      "relevance": 0.542741,
      "count": 1
    },
    {
      "text": "Guerbet’s first project",
      "relevance": 0.542725,
      "count": 1
    },
    {
      "text": "common site",
      "relevance": 0.542448,
      "count": 1
    },
    {
      "text": "exclusive joint development agreement",
      "relevance": 0.540924,
      "count": 1
    },
    {
      "text": "IBM Research team",
      "relevance": 0.537311,
      "count": 1
    },
    {
      "text": "secondary liver cancer",
      "relevance": 0.53605,
      "count": 1
    },
    {
      "text": "United States",
      "relevance": 0.531469,
      "count": 2
    },
    {
      "text": "Guerbet’s CEO",
      "relevance": 0.530873,
      "count": 1
    },
    {
      "text": "diagnostic support tool",
      "relevance": 0.530526,
      "count": 1
    },
    {
      "text": "liver disease",
      "relevance": 0.530347,
      "count": 1
    },
    {
      "text": "second leading cause of cancer death",
      "relevance": 0.53003,
      "count": 1
    },
    {
      "text": "innovative solutions",
      "relevance": 0.529917,
      "count": 1
    },
    {
      "text": "Vice President",
      "relevance": 0.528412,
      "count": 1
    },
    {
      "text": "medical devices",
      "relevance": 0.528087,
      "count": 1
    },
    {
      "text": "IBM Watson image analytics",
      "relevance": 0.527109,
      "count": 2
    },
    {
      "text": "fastest growing cause of cancer deaths",
      "relevance": 0.525308,
      "count": 1
    },
    {
      "text": "comprehensive range of pharmaceutical products",
      "relevance": 0.525246,
      "count": 1
    },
    {
      "text": "growing health concern",
      "relevance": 0.52488,
      "count": 1
    },
    {
      "text": "significant respective expertise of IBM",
      "relevance": 0.524304,
      "count": 1
    },
    {
      "text": "primary liver cancer cases",
      "relevance": 0.522918,
      "count": 1
    },
    {
      "text": "agreement aims",
      "relevance": 0.522794,
      "count": 1
    },
    {
      "text": "contrast agents",
      "relevance": 0.521267,
      "count": 1
    },
    {
      "text": "therapy prediction",
      "relevance": 0.520061,
      "count": 1
    },
    {
      "text": "better characterization",
      "relevance": 0.519191,
      "count": 1
    },
    {
      "text": "drug discovery",
      "relevance": 0.518761,
      "count": 1
    },
    {
      "text": "informed characterizations of tissue",
      "relevance": 0.518079,
      "count": 1
    },
    {
      "text": "care",
      "relevance": 0.516453,
      "count": 2
    },
    {
      "text": "challenge of liver cancer",
      "relevance": 0.51404,
      "count": 1
    },
    {
      "text": "business unit of IBM",
      "relevance": 0.513358,
      "count": 1
    },
    {
      "text": "Liver",
      "relevance": 0.512745,
      "count": 4
    },
    {
      "text": "use",
      "relevance": 0.511526,
      "count": 1
    },
    {
      "text": "France",
      "relevance": 0.511378,
      "count": 2
    },
    {
      "text": "collaboration",
      "relevance": 0.510413,
      "count": 2
    },
    {
      "text": "solutions",
      "relevance": 0.510403,
      "count": 1
    },
    {
      "text": "part of a family of decision support tools",
      "relevance": 0.508694,
      "count": 1
    },
    {
      "text": "workflows of healthcare professionals",
      "relevance": 0.507926,
      "count": 1
    },
    {
      "text": "revenue",
      "relevance": 0.507672,
      "count": 2
    },
    {
      "text": "offerings",
      "relevance": 0.507515,
      "count": 1
    }
  ],
  "entities": [
    {
      "type": "Company",
      "text": "IBM Watson Health",
      "relevance": 0.892684,
      "count": 9
    },
    {
      "type": "HealthCondition",
      "text": "liver cancer",
      "relevance": 0.617521,
      "disambiguation": {
        "subtype": [
          "DiseaseOrMedicalCondition",
          "CauseOfDeath",
          "Disease"
        ],
        "name": "Hepatocellular carcinoma",
        "dbpedia_resource": "http://dbpedia.org/resource/Hepatocellular_carcinoma"
      },
      "count": 6
    },
    {
      "type": "Company",
      "text": "Guerbet",
      "relevance": 0.577379,
      "disambiguation": {
        "subtype": [],
        "name": "Guerbet",
        "dbpedia_resource": "http://dbpedia.org/resource/Guerbet"
      },
      "count": 12
    },
    {
      "type": "JobTitle",
      "text": "Watson Imaging Care Advisor for Liver",
      "relevance": 0.477049,
      "count": 1
    },
    {
      "type": "Company",
      "text": "IBM",
      "relevance": 0.429494,
      "disambiguation": {
        "subtype": [
          "SoftwareLicense",
          "OperatingSystemDeveloper",
          "ProcessorManufacturer",
          "SoftwareDeveloper",
          "CompanyFounder",
          "ProgrammingLanguageDesigner",
          "ProgrammingLanguageDeveloper"
        ],
        "name": "IBM",
        "dbpedia_resource": "http://dbpedia.org/resource/IBM"
      },
      "count": 5
    },
    {
      "type": "Company",
      "text": "IBM Watson",
      "relevance": 0.424247,
      "count": 2
    },
    {
      "type": "JobTitle",
      "text": "Watson Imaging Care Advisor",
      "relevance": 0.388068,
      "count": 1
    },
    {
      "type": "Company",
      "text": "Watson Health technologies",
      "relevance": 0.296781,
      "count": 1
    },
    {
      "type": "HealthCondition",
      "text": "liver disease",
      "relevance": 0.275794,
      "disambiguation": {
        "subtype": [
          "DiseaseOrMedicalCondition",
          "DiseaseCause",
          "RiskFactor",
          "Disease"
        ],
        "name": "Liver disease",
        "dbpedia_resource": "http://dbpedia.org/resource/Liver_disease"
      },
      "count": 1
    },
    {
      "type": "Organization",
      "text": "AI",
      "relevance": 0.208705,
      "disambiguation": {
        "subtype": [],
        "name": "The Art Institutes",
        "dbpedia_resource": "http://dbpedia.org/resource/The_Art_Institutes"
      },
      "count": 1
    },
    {
      "type": "JobTitle",
      "text": "Vice President of Imaging",
      "relevance": 0.206762,
      "count": 1
    },
    {
      "type": "Location",
      "text": "France",
      "relevance": 0.206536,
      "disambiguation": {
        "subtype": [
          "Region",
          "AdministrativeDivision",
          "GovernmentalJurisdiction",
          "FilmDirector",
          "Country"
        ],
        "name": "France",
        "dbpedia_resource": "http://dbpedia.org/resource/France"
      },
      "count": 2
    },
    {
      "type": "Person",
      "text": "Villepinte",
      "relevance": 0.191729,
      "count": 1
    },
    {
      "type": "Organization",
      "text": "USA",
      "relevance": 0.184507,
      "disambiguation": {
        "subtype": [
          "Location",
          "HumanLanguage",
          "PoliticalDistrict",
          "Region",
          "AdministrativeDivision",
          "Country",
          "GovernmentalJurisdiction"
        ],
        "name": "Italy",
        "dbpedia_resource": "http://dbpedia.org/resource/Italy"
      },
      "count": 1
    },
    {
      "type": "Location",
      "text": "Cambridge",
      "relevance": 0.18094,
      "disambiguation": {
        "subtype": [
          "City"
        ]
      },
      "count": 1
    },
    {
      "type": "Person",
      "text": "Jessica Emond",
      "relevance": 0.178213,
      "count": 1
    },
    {
      "type": "JobTitle",
      "text": "Advisor",
      "relevance": 0.176557,
      "count": 1
    },
    {
      "type": "Person",
      "text": "Yves L’Epine",
      "relevance": 0.160057,
      "count": 1
    },
    {
      "type": "Location",
      "text": "United States.",
      "relevance": 0.156519,
      "disambiguation": {
        "subtype": [
          "Country"
        ]
      },
      "count": 1
    },
    {
      "type": "Person",
      "text": "François Nicolas",
      "relevance": 0.15514,
      "count": 1
    },
    {
      "type": "Company",
      "text": "Euronext Paris",
      "relevance": 0.154172,
      "count": 1
    },
    {
      "type": "Person",
      "text": "Anne Le Grand",
      "relevance": 0.153979,
      "count": 1
    },
    {
      "type": "Company",
      "text": "NYSE",
      "relevance": 0.153158,
      "count": 1
    },
    {
      "type": "JobTitle",
      "text": "CEO",
      "relevance": 0.152276,
      "count": 1
    },
    {
      "type": "JobTitle",
      "text": "Chief Digital Officer",
      "relevance": 0.149375,
      "count": 1
    },
    {
      "type": "Location",
      "text": "United States",
      "relevance": 0.137907,
      "disambiguation": {
        "subtype": [
          "Region",
          "AdministrativeDivision",
          "GovernmentalJurisdiction",
          "FilmEditor",
          "Country"
        ],
        "name": "United States",
        "dbpedia_resource": "http://dbpedia.org/resource/United_States"
      },
      "count": 1
    },
    {
      "type": "Location",
      "text": "Israel",
      "relevance": 0.130299,
      "disambiguation": {
        "subtype": [
          "GovernmentalJurisdiction",
          "FilmArtDirector",
          "Country"
        ],
        "name": "Israel",
        "dbpedia_resource": "http://dbpedia.org/resource/Israel"
      },
      "count": 1
    },
    {
      "type": "EmailAddress",
      "text": "Jessica.emond@ibm.com",
      "relevance": 0.130299,
      "count": 1
    },
    {
      "type": "Quantity",
      "text": "83 percent",
      "relevance": 0.130299,
      "count": 1
    },
    {
      "type": "Quantity",
      "text": "90 years",
      "relevance": 0.130299,
      "count": 1
    },
    {
      "type": "Quantity",
      "text": "50%",
      "relevance": 0.130299,
      "count": 1
    },
    {
      "type": "Quantity",
      "text": "8%",
      "relevance": 0.130299,
      "count": 1
    }
  ],
  "concepts": [
    {
      "text": "Cancer",
      "relevance": 0.984368,
      "dbpedia_resource": "http://dbpedia.org/resource/Cancer"
    },
    {
      "text": "Health care",
      "relevance": 0.885532,
      "dbpedia_resource": "http://dbpedia.org/resource/Health_care"
    },
    {
      "text": "Magnetic resonance imaging",
      "relevance": 0.66221,
      "dbpedia_resource": "http://dbpedia.org/resource/Magnetic_resonance_imaging"
    },
    {
      "text": "Medicine",
      "relevance": 0.657648,
      "dbpedia_resource": "http://dbpedia.org/resource/Medicine"
    },
    {
      "text": "Artificial intelligence",
      "relevance": 0.595164,
      "dbpedia_resource": "http://dbpedia.org/resource/Artificial_intelligence"
    },
    {
      "text": "Oncology",
      "relevance": 0.571111,
      "dbpedia_resource": "http://dbpedia.org/resource/Oncology"
    },
    {
      "text": "Liver",
      "relevance": 0.563501,
      "dbpedia_resource": "http://dbpedia.org/resource/Liver"
    },
    {
      "text": "Hepatocellular carcinoma",
      "relevance": 0.563287,
      "dbpedia_resource": "http://dbpedia.org/resource/Hepatocellular_carcinoma"
    }
  ],
  "categories": [
    {
      "score": 0.955915,
      "label": "/health and fitness"
    },
    {
      "score": 0.945541,
      "label": "/health and fitness/therapy"
    },
    {
      "score": 0.875784,
      "label": "/health and fitness/disease/cancer/brain tumor"
    }
```

https://movie.naver.com/movie/bi/mi/point.nhn?code=185917 

```shell
curl -X POST -u "apikey:yourkey" \
"yourURL" \
--header "Content-Type: application/json" \
--data '{
  "url": "https://movie.naver.com/movie/bi/mi/point.nhn?code=185917",
  "features": {
    "sentiment": {},
    "categories": {},
    "concepts": {},
    "entities": {},
       "keywords": {
      "emotion": true
    }
  }
}'
```

"반도"라는 영화 텍스트 분석이다... 좋아보이진 않는다.

```shell
{
  "usage": {
    "text_units": 1,
    "text_characters": 1704,
    "features": 5
  },
  "sentiment": {
    "document": {
      "score": 0.30744,
      "label": "positive"
    }
  },
  "retrieved_url": "https://movie.naver.com/movie/bi/mi/point.nhn?code=185917",
  "language": "ko",
  "keywords": [
    {
      "text": "부산행",
      "relevance": 0.675291,
      "count": 6
    },
    {
      "text": "캐릭터들",
      "relevance": 0.590918,
      "count": 1
    },
    {
      "text": "vs인간",
      "relevance": 0.576187,
      "count": 1
    },
    {
      "text": "강동원",
      "relevance": 0.556464,
      "count": 1
    },
    {
      "text": "액션의 핵심",
      "relevance": 0.554405,
      "count": 1
    },
    {
      "text": "장면들",
      "relevance": 0.550989,
      "count": 1
    },
    {
      "text": "마동석의 완력",
      "relevance": 0.543865,
      "count": 1
    },
    {
      "text": "오락 영화",
      "relevance": 0.539792,
      "count": 1
    },
    {
      "text": "전반부 빈틈",
      "relevance": 0.538315,
      "count": 1
    },
    {
      "text": "총기 액션",
      "relevance": 0.536781,
      "count": 1
    },
    {
      "text": "액션물",
      "relevance": 0.535982,
      "count": 1
    },
    {
      "text": "어느 정도 효과",
      "relevance": 0.535609,
      "count": 1
    },
    {
      "text": "뿐 결국 사람",
      "relevance": 0.529437,
      "count": 1
    },
    {
      "text": "이야기의 중심",
      "relevance": 0.527942,
      "count": 1
    },
    {
      "text": "디스토피아의 특정 부분",
      "relevance": 0.527572,
      "count": 1
    },
    {
      "text": "관객의 눈물",
      "relevance": 0.527452,
      "count": 1
    },
    {
      "text": "이정현",
      "relevance": 0.52636,
      "count": 1
    },
    {
      "text": "강동원의 미모",
      "relevance": 0.526125,
      "count": 1
    },
    {
      "text": "자신의 존재",
      "relevance": 0.524436,
      "count": 1
    },
    {
      "text": "여자들",
      "relevance": 0.52181,
      "count": 1
    },
    {
      "text": "극 전체",
      "relevance": 0.520471,
      "count": 1
    },
    {
      "text": "장르적 쾌감",
      "relevance": 0.520054,
      "count": 1
    },
    {
      "text": "전체 풍경",
      "relevance": 0.519577,
      "count": 1
    },
    {
      "text": "색다른 액션",
      "relevance": 0.519245,
      "count": 1
    },
    {
      "text": "시각적 경험",
      "relevance": 0.517534,
      "count": 1
    },
    {
      "text": "폐허 대한민국",
      "relevance": 0.516584,
      "count": 1
    },
    {
      "text": "생존자들의 사투",
      "relevance": 0.515978,
      "count": 1
    },
    {
      "text": "사람 사이의 이야기",
      "relevance": 0.515242,
      "count": 1
    },
    {
      "text": "총격전",
      "relevance": 0.512915,
      "count": 1
    },
    {
      "text": "아이들",
      "relevance": 0.512629,
      "count": 1
    },
    {
      "text": "인천항의 모습",
      "relevance": 0.512206,
      "count": 1
    },
    {
      "text": "극의 진행",
      "relevance": 0.511018,
      "count": 1
    },
    {
      "text": "서울 곳곳",
      "relevance": 0.511017,
      "count": 1
    },
    {
      "text": "이예원",
      "relevance": 0.510392,
      "count": 1
    },
    {
      "text": "인물 개개인의 사연",
      "relevance": 0.510197,
      "count": 1
    },
    {
      "text": "다양한 총기",
      "relevance": 0.509567,
      "count": 1
    },
    {
      "text": "고난도 카",
      "relevance": 0.509316,
      "count": 1
    },
    {
      "text": "순간적 파괴력",
      "relevance": 0.50899,
      "count": 1
    },
    {
      "text": "바이러스 발생",
      "relevance": 0.50788,
      "count": 1
    },
    {
      "text": "액션",
      "relevance": 0.507726,
      "count": 1
    },
    {
      "text": "전작",
      "relevance": 0.507219,
      "count": 3
    },
    {
      "text": "신선하다는 인상",
      "relevance": 0.506369,
      "count": 1
    },
    {
      "text": "공간의 분위기",
      "relevance": 0.505,
      "count": 1
    },
    {
      "text": "첫 영화",
      "relevance": 0.504722,
      "count": 1
    },
    {
      "text": "연상호 감독 특유의 디스토피아적 세계관",
      "relevance": 0.504572,
      "count": 1
    },
    {
      "text": "희귀하다는 인상",
      "relevance": 0.504285,
      "count": 1
    },
    {
      "text": "연약한지 야",
      "relevance": 0.503985,
      "count": 1
    },
    {
      "text": "익숙한 공간",
      "relevance": 0.503593,
      "count": 1
    },
    {
      "text": "독립영화",
      "relevance": 0.502713,
      "count": 1
    },
    {
      "text": "확실하게 각인",
      "relevance": 0.502694,
      "count": 1
    }
  ],
  "entities": [
    {
      "type": "Person",
      "text": "연상호 감독",
      "relevance": 0.952545,
      "count": 2
    },
    {
      "type": "Person",
      "text": "마동석",
      "relevance": 0.565046,
      "count": 1
    },
    {
      "type": "Person",
      "text": "이예원",
      "relevance": 0.461565,
      "count": 1
    },
    {
      "type": "Person",
      "text": "헐거운",
      "relevance": 0.441754,
      "count": 1
    },
    {
      "type": "Person",
      "text": "강동원",
      "relevance": 0.421448,
      "count": 2
    },
    {
      "type": "Person",
      "text": "감독 연상호",
      "relevance": 0.389125,
      "count": 1
    },
    {
      "type": "Person",
      "text": "이정현",
      "relevance": 0.3578,
      "count": 1
    },
    {
      "type": "Person",
      "text": "이레",
      "relevance": 0.336244,
      "count": 1
    },
    {
      "type": "Location",
      "text": "대한민국",
      "relevance": 0.179842,
      "count": 1
    },
    {
      "type": "Quantity",
      "text": "4년",
      "relevance": 0.15146,
      "count": 1
    },
    {
      "type": "Location",
      "text": "인천",
      "relevance": 0.123408,
      "count": 1
    },
    {
      "type": "Quantity",
      "text": "구도",
      "relevance": 0.121433,
      "count": 1
    },
    {
      "type": "Location",
      "text": "서울",
      "relevance": 0.102665,
      "disambiguation": {
        "subtype": [
          "AdministrativeDivision",
          "OlympicHostCity",
          "CityTown"
        ],
        "name": "서울특별시",
        "dbpedia_resource": "http://ko.dbpedia.org/resource/서울특별시"
      },
      "count": 1
    },
    {
      "type": "Location",
      "text": "한국",
      "relevance": 0.073895,
      "count": 1
    }
  ],
  "concepts": [
    {
      "text": "사람",
      "relevance": 0.993985,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/사람"
    },
    {
      "text": "감정",
      "relevance": 0.939951,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/감정"
    },
    {
      "text": "독립 영화",
      "relevance": 0.742893,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/독립_영화"
    },
    {
      "text": "기쁨",
      "relevance": 0.701024,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/기쁨"
    },
    {
      "text": "여성",
      "relevance": 0.689057,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/여성"
    },
    {
      "text": "대한민국",
      "relevance": 0.652193,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/대한민국"
    },
    {
      "text": "김구",
      "relevance": 0.602916,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/김구"
    },
    {
      "text": "플롯",
      "relevance": 0.507928,
      "dbpedia_resource": "http://ko.dbpedia.org/resource/플롯"
    }
  ],
  "categories": [
    {
      "score": 0.922816,
      "label": "/art and entertainment/movies and tv/movies"
    },
    {
      "score": 0.827163,
      "label": "/art and entertainment/movies and tv/movies/reviews"
    },
    {
      "score": 0.819309,
      "label": "/art and entertainment/movies and tv/action"
    }
  ]
}%
```

이상으로 간단히 IBM cloud AI API를 확인해보았다.