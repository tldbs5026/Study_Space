# args
- argument

# criterion
- BCEWithLogitsLoss

# dataloader

## preprocess
- get_train, test
- split
- cate_cols를 받아서
- LE 진행
- convert_time : 시간 형식을 int로 바굼
- __feature_engineering : 채워야 할 부분
- arg를 받아서 sorting value

## DKTDataset
- test, question, tag, correct를 받아서
- masking
- interaction
- loader를 통해 masking한 값을 train, valset으로 DataLoading

