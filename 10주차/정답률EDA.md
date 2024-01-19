
knowledgeTag를 기준으로 나온 비율


explicit
- 문항의 정답률
- 문항의 Tag의 정답률

```
df.groupby('knowledgeTag')['answerCode']
```

implicit
- embedding

# 가설들

## h1 : 유저가 푼 문항 수가 많을수록 정답률은 높아질 것이다.
- 여러문제를 많이 푸는 것으로 인한 학습효과로 인해 실력이 향상할 것이라는 가정.

```
stu_groupby = train.groupby('userID').agg({
    'assessmentItemID': 'count',
    'answerCode': percentile
})
```

```
corr_result = stu_groupby.corr()
corrlation = corr_result.loc['assessmentItemID']['answerCode']
p_value = pearsonr(stu_groupby['assessmentItemID'], stu_groupby['answerCode'])[1]
```
[plot](https://colab.research.google.com/drive/1zOsihu1mya9o88g5hVWFOh_3bpQvx29Y#scrollTo=Fop5BgTiyLOC)
- 문항수가 많을수록 분포상으로 정답률이 증가하는 것으로 나타났다.
- 추가적으로, 상관분석 결과, r=0.17(p=0.000)으로 나타났다.

## h1 : 문제의 태그와 정답률은 양의 상관을 보일 것이다.
- 태그가 많이 노출될수록 학습자들이 이에 대해 많은 시도를 할 것이며, 그에 따른 정답률은 그렇지 않은 경우보다 높아질 것이라 가정하였다.

```
tag_groupby = train.groupby('KnowledgeTag').agg({
    'userID': 'count',
    'answerCode': percentile
})

corr_result = tag_groupby.corr()

corrlation = corr_result.loc['userID']['answerCode']
p_value = pearsonr(tag_groupby['userID'], tag_groupby['answerCode'])[1]
```
- 상관분석 결과, 노출이 많이된 Tag와 정답률간의 상관은 0.376(p=0.000)으로 나타났다.

## h1 : 누적된 문제수가 많아질수록 정답률은 높아질 것이다.


## h1 : 문제지의 번호가 높아질수록 어려운 문제일 것이다.
- 번호가 높아짐에 따라 정답률이 낮아져야함.

```
problem = train.copy()
problem['problem'] = problem['assessmentItemID'].str[7:].astype(int)
correlation, p_value = pearsonr(problem['problem'], problem['answerCode'])
```
- 상관 분석 결과, 문제의 번호와 정답률은 r=-0.166(p=0.0)으로 약한 음의 상관을 보이는 것으로 확인.

## h1 : 문제가 진행됨에 따라 난이도는 어려워 질 것이다.
	문제지는 쉬운난이도부터 어려운 난이도로 진행되어 변별력을 가지는 것이 일반적이다. 따라서 본 데이터에서의 문제집도 유형과 관계없이 문제가 진행됨에 따라 어려운 난이도를 보일 것이며, 낮은 정답률을 보일 것으로 예상할 수 있다.

```
df = train.copy()
# assessmentItemID의 뒤에 3자리를 의미 -> 각 시험지 별로 문제번호
df['item'] = df['assessmentItemID'].apply(lambda x: x[7:]) 
# item 열을 int16으로 변경
df["item"] = df["item"].astype("int16")
```

```
# test 별로 그룹화 한뒤 각 test 별로 마지막 문항을 나타내는 last_prob_no 피쳐 생성
tmp = df.groupby(["testId"]).agg({"item": "max"})
tmp.rename(columns={"item": "last_prob_no"}, inplace=True)

# 기존의 DF와 merge 하여 각각의 시험지의 마지막 문항을 나타내는 피쳐 추가
tmp_df = df.copy()
tmp_df = pd.merge(tmp_df, tmp, on="testId", how="left")

# 마지막 문제와 가까운 정도를 계산
tmp_df["last_prob"] = tmp_df["item"] / tmp_df["last_prob_no"]
tmp_df = tmp_df.drop("last_prob_no", axis=1)
tmp_df.head()
```
![[Pasted image 20240108140104.png]]
- 분포상으로는 우하향하는 것으로 확인되었다.

```
corr_result= last_prob_group.corr()

correlation = corr_result.loc['last_prob','answerCode']
p_value = pearsonr(last_prob_group['last_prob'], last_prob_group['answerCode'])[1]
```
- 상관분석 결과, 문제의 마지막과 정답률은 r=-0.447(p=0.000)으로 나타났다. 문제가 마지막에 가까워질 수록 정답률과는 음의 상관을 보여, 실제로 문제가 진행됨에 따라 어려워지는 구조라고 해석할 수 있다.
- 파악한 결과는 다음과 같다.
```
# 누적합
_cumsum = train.loc[:, ['userID', 'answerCode']].groupby('userID').agg({'answerCode': 'cumsum'})
# 누적갯수
_cumcount = train.loc[:, ['userID', 'answerCode']].groupby('userID').agg({'answerCode': 'cumcount'}) + 1

cum_ans = _cumsum / _cumcount
train['cumsum'] = cum_ans['answerCode']

train['item_different'] = train['assessmentItemID'].apply(lambda x: x[7:]) # assessmentItemID의 뒤에 3자리를 의미 -> 각 시험지 별로 문제번호
# item 열을 int16으로 변경
train["item_different"] = train["item_different"].astype("int16")

train.groupby('item_different')['answerCode'].agg(avg_percent)
```
```
item_different
1     0.749916
2     0.720062
3     0.687773
4     0.663364
5     0.599134
6     0.555685
7     0.515399
8     0.457156
9     0.481729
10    0.527892
11    0.480609
12    0.370370
```
- 어려운 문제를 푸는 것은 기본적으로 잘하는 학생일 가능성이 높다.
- 이를 고려하여 실제 잘하는 그룹과 못하는 그룹을 나누어서 확인해 보아야 할 필요가 있음.

KnowledgeTag에 따른  cumsum의 결과는 다음과 같다.
```
train.groupby('KnowledgeTag')['cumsum'].agg(avg_percent).sort_values()

KnowledgeTag
4234     0.533051
10590    0.533274
9745     0.535641
9744     0.536943
10169    0.537421
           ...   
5844     0.761138
7226     0.769503
7271     0.771203
7225     0.777803
7224     0.786134
```
- memo siyun : 이론상으로 각각의 유형에 해당하는 문제는 지식이 누적됨(많이 풀어볼수록) 정답률이 늘어나는 구조라고 예상할 수 있다.
	- 이 부분은 cumsum을 이용하지 않고 단순 count와의 차이를 보는 것이 맞다고 생각.
- 태그 자체가 낮은 정답률을 가질 경우를 따로 뽑아서 분석해볼 것.

