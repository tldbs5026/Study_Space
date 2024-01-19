
1. 임베딩 준비



```
num_users = len(userid2index)
        user_emb = embeddings[:num_users]
        item_emb = embeddings[num_users: ]
        torch.save(user_emb, os.path.join(model_dir, 'user_emb.pt'))
        torch.save(item_emb, os.path.join(model_dir, 'item_emb.pt'))
```
임베딩을 user, item과 분리해야함.
user와 item의 상호작용을 기반으로 학습되었기 때문에, 하나의 좌표 그래프보다는 각각의 user, item이 있는 임베딩 좌표로 지정해줘야 의미가 있음.

user_emb = torch.load(path + 'user_emb.pt')
item_emb = torch.load(path + 'item_emb.pt')

# 수정해야하는 부분
## dataloader.py 예시
```
class DKTDataset(torch.utils.data.Dataset):

    def __init__(self, data: np.ndarray, args, user_embedding=None, item_embedding=None):
        self.data = data
        self.max_seq_len = args.max_seq_len
        self.args = args
        #feat siyun - add embedding
        self.user_embedding = user_embedding
		self.item_embedding = item_embedding

	def __getitem__(self, index  :int) -> dict :
		row = self.data[index]
		...
		...
		...
		
        data['user_embedding'] = self.user_embedding[userid]
        data['item_embedding'] = self.item_embedding[itemid]
		
```

```
def get_loaders(args, train: np.ndarray, valid: np.ndarray, user_emb, item_emb)

trainset = DKTDataset(train, args, user_emb, item_emb)
```

추가적 피드백 : 임베딩을 여기에 받아놓고 modelbase에 임베딩을 concat할 때 다시 로딩할 것.



## model.py

```
class lstm() :

	def forward(self, test, question, tag, corre,t ~~  user_embedding, item_embedding)

	combined_emb = torch.cat([user_emb, item_emb, input1, input2,~~], dim-=1)
	
	return out
```





userid 필드는 실제 사용자의 고유 id값을 가지고 있음.
따라서 순차적이지 않을 수 있으며, 임베딩 조회를 위해 각 id값을 0부터 시작하는 연속적인 인덱스로 변환해야함.

임베딩 좌표와 유저, 아이템을 일일히 일치시켜줘야함.

각각의 shape은 
torch.Size([7442, 64]) ,torch.Size([9454, 64])

[0] : 유저, 아이템의 고유번호
[1] : 유저, 아이템에 대한 설명

dataloader.py에서 수정한 부분
```
# feat siyun : add embedding

        self.user_emb = torch.load('/data/ephemeral/siyun_lightgcn_2_fm/models/user_emb.pt')
        self.item_emb = torch.load('/data/ephemeral/siyun_lightgcn_2_fm/models/item_emb.pt')
        ~~~
        # feat siyun : preprocessing에서 임베딩을 처리했기 때문에 여기에서 추가적으로 받도록 data 추가.

        data['user_idx'] = torch.tensor(row['user_idx'])

        data['item_idx'] = torch.tensor(row['item_idx'])
```


model.py

forward에서 수정한 부분
```
def forward(self, data, user_emb, item_emb):
        interaction = data["interaction"]
        batch_size = interaction.size(0)
        # [찬우] seq_len 추가
        seq_len = interaction.size(1)

        # feat siyun : add user_emb, item_emb in total embed
        ## ---------------
        user_emb_batch = self.user_emb[data['user_idx']].view(batch_size, seq_len, -1)
        item_emb_batch = self.item_emb[data['item_idx']].view(batch_size, seq_len, -1)
        ## ---------------

        ####### [건우] Embedding + concat ######  
        # category embeding + concat
        embed_interaction = self.embedding_interaction(interaction.int())

        embed_cat_feats = []
        for cat_col in self.args.cat_cols:
            value = data[cat_col]
            embed_cat_feat = getattr(self, f'embedding_{cat_col}')(value.int()) # self.embedding_xxx(xxx.int())
            embed_cat_feats.append(embed_cat_feat)

        # feat siyun : concat embedding with user,item embedding
        ## 임베딩을 사용한다면 이 옵션 진행.
        embed = torch.cat([embed_interaction,*embed_cat_feats, user_emb_batch, item_emb_batch], dim=2)
        # embed = torch.cat([embed_interaction,*embed_cat_feats],dim=2) # dim=2는 3차원을 합친다는 의미

        # [승준] encoder embed 추가
        enc_embed = torch.cat([*embed_cat_feats],dim=2)
        # continious concat
        con_feats = []
        for con_col in self.args.con_cols:
            value = data[con_col]
            con_feats.append(value.unsqueeze(2))
        embed = torch.cat([embed,*con_feats], dim=2).float()

        ################# [건우] ###############
        # [승준] encoder embed concat
        enc_embed = torch.cat([enc_embed,*con_feats], dim=2).float()

        X = self.comb_proj(embed) # concat후 feature_size=Hidden dimension으로 선형변환

        # [승준] encoder embed proj 추가
        enc_X = self.enc_comb_proj(enc_embed)
        # [찬우] LastQuery 모델의 positional_encoding을 위한 seq_len 추가
        return enc_X, X, batch_size, seq_len # embedding을 concat하고 선형 변환한 값
```


```
user_emb_batch = self.user_emb[data['user_idx']].view(batch_size, seq_len, -1)
        item_emb_batch = self.item_emb[data['item_idx']].view(batch_size, seq_len, -1)
```
- 배치 및 시퀀스 차원 조정