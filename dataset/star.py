import torch
from torch.utils.data import Dataset
import json

class STAR(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        self.split = split
        self.data = json.load(open(f'./data/star/STAR_{split}.json', 'r'))
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
        self.num_options = 4
        print(f"Num {split} data: {len(self.data)}") 


    def _get_text(self, idx):
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
            
        options = {x['choice_id']: x['choice'] for x in self.data[idx]['choices']}
        options = [options[i] for i in range(self.num_options)]
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        if self.split!="test":
            answer = options.index(self.data[idx]['answer'])
            return text, answer
        else:
            return text, 0


    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        qtype = self.qtype_mapping[self.data[idx]['question_id'].split('_')[0]]
        # if self.split!="test":
        text, answer = self._get_text(idx)
        # text_id, label, video_start, video_index, label_mask = [], [], [], [], []
        # for i in answer:
        #     w, x, y, z, t =self._get_text_token(text, i)
        #     text_id.append(w)
        #     label.append(x)
        #     video_start.append(y)
        #     video_index.append(z)
        #     label_mask.append(t)
        # start, end = round(self.data[idx]['start']), round(self.data[idx]['end'])
        # video, video_len = self._get_video(f'{vid}', start, end)
        # return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
        #         "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype, "qid":self.data[idx]['question_id']}

        return text, self.data[idx]['question_id'], answer
        # else:
        #     text = self._get_text(idx)
        #     start, end = round(self.data[idx]['start']), round(self.data[idx]['end'])
        #     video, video_len = self._get_video(f'{vid}', start, end)
        #     return {"vid": vid, "video": video, "video_len": video_len, "text": text, "qid": idx, "qtype": qtype, "qid":self.data[idx]['question_id']}
    def __len__(self):
        return len(self.data)