https://www.kaggle.com/code/erikbruin/riiid-comprehensive-eda-baseline/notebook
다른 eda 참고
https://github.com/boostcampaitech5/level2_dkt-recsys-09/blob/main/eda/eda_integration.ipynb

시퀀스 모델링
- agg 진행 후 Feature Engineering
- transaction 그대로 사용한 후, Featrue Engineering
![[Pasted image 20240104103048.png]]
![[Pasted image 20240104103054.png]]

# 데이터의 구조
## userID
- 사용자 별 고유번호
- 총 7,442명의 unique
## assessmentitemID
- 사용자가 푼 문항의 일련 번호
- 9,454개의 unique
- 규칙
    - 첫 자리는 A
    - 그 다음 6자리 시험지 번호
    - 마지막 3자리는 시험지 내 문항의 번호
	    
## testID
- 시험지의 일련 번호
- 1537개의 unique
- 규칙
    - 첫 자리는 A
    - 첫 자리 3개 + 마지막 3개 는 시험지 번호
    - 가운데는 날려도 괜찮음.
    - 앞의 3자리에서 가운데만 1~9
	    - 3자리에 해당하는 category 칼럼을 따로 만들어 추후 활용 가능성 존재.
## answerCode
- 정답/오답 여부
- 정답이 약 65%
- 사람의 실력이 중요하다면
	- cumsum을 이용해서 새로운 FE 칼럼 생성
- 문제의 난이도가 중요하다면
	- 문제의 정답율을 사용

## Timestamp
- 사용자가 interaction를 시작한 시간
```
diff_train = train.loc[:, ['userID','Timestamp']].groupby('userID').diff().shift(-1)

diff_train = diff_train['Timestamp'].apply(lambda x : x.total_seconds())

train['elapsed'] = diff_train
```
- 시간차가 많이 나는 경우에 처리를 어떻게 해야할지?

## KnowleadgeTag
- 문항의 중분류
- 912개의 unique


data split
- 이벤트 별로 묶는 것이 아닌 사용자 별로 묶어서 동질성 유지
- sequential한 데이터는 각각의 userID를 기준으로 나누어야 함을 의미

Training
- 하이퍼 파라미터는 마지막
- FE를 가장 먼저


Task에 따른 입출력 구조
![[Pasted image 20240104103559.png]]
one to one
- 입력이 출력으로

one to many
- 입력을 받아 extract한 후 새로운 임베딩

Many to One
- 시퀀스 입력, 결과 예측
- 신용카드 기록으로 성별 예측과 같은 문제

Many to Many
- bert와 같은 구조